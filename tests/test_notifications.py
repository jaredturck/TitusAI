import json
import threading

import notifications
from notifications import DiscordNotifier, EMBED_COLORS, format_embed
from train import (
    format_duration,
    format_progress_bar,
    training_status_description,
    training_status_fields,
)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return b''


def make_config():
    return {
        'enabled': True,
        'username': 'TitusAI Test',
        'status_interval_seconds': 600,
        'request_timeout_seconds': 1,
    }


def test_discord_notifier_posts_embed(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(notifications, 'urlopen', fake_urlopen)
    monkeypatch.setenv('STATUS_WEBHOOK', 'https://discord.com/api/webhooks/test/token')

    notifier = DiscordNotifier(make_config())
    notifier.send(
        '📈 Training progress',
        [('Step', '1'), ('Loss', '5.0000')],
        description='`████░░░░` 50.00% complete',
        color='purple',
    )

    assert notifier.close()
    assert len(requests) == 1

    request, timeout = requests[0]
    payload = json.loads(request.data.decode('utf-8'))
    embed = payload['embeds'][0]

    assert timeout == 1
    assert payload['username'] == 'TitusAI Test'
    assert payload['allowed_mentions'] == {'parse': []}
    assert 'content' not in payload
    assert embed['title'] == '📈 Training progress'
    assert embed['description'] == '`████░░░░` 50.00% complete'
    assert embed['color'] == EMBED_COLORS['purple']
    assert embed['fields'][0] == {
        'name': 'Step',
        'value': '1',
        'inline': True,
    }
    assert 'timestamp' in embed
    assert 'footer' in embed



def test_embed_formatting_runs_on_background_thread(monkeypatch):
    format_threads = []
    original_format_embed = notifications.format_embed

    def tracked_format_embed(**event):
        format_threads.append(threading.current_thread().name)
        return original_format_embed(**event)

    monkeypatch.setattr(notifications, 'format_embed', tracked_format_embed)
    monkeypatch.setattr(notifications, 'urlopen', lambda request, timeout: FakeResponse())
    monkeypatch.setenv('STATUS_WEBHOOK', 'https://discord.com/api/webhooks/test/token')

    notifier = DiscordNotifier(make_config())
    notifier.send('Training progress', [('Loss', '4.0000')])

    assert notifier.close()
    assert format_threads == ['titus-discord-notifier']

def test_missing_webhook_disables_notifier(monkeypatch, tmp_path):
    monkeypatch.delenv('STATUS_WEBHOOK', raising=False)
    monkeypatch.setattr(notifications, 'PROJECT_ROOT', tmp_path)

    notifier = DiscordNotifier(make_config())

    assert not notifier.enabled
    assert notifier.close()


def test_notification_failure_does_not_raise(monkeypatch):
    def failed_urlopen(request, timeout):
        raise TimeoutError('timed out')

    monkeypatch.setattr(notifications, 'urlopen', failed_urlopen)
    monkeypatch.setenv('STATUS_WEBHOOK', 'https://discord.com/api/webhooks/test/token')

    notifier = DiscordNotifier(make_config())
    notifier.send('Training status', [('Step', '50')])

    assert not notifier.close()
    assert notifier.last_error == 'timed out'


def test_webhook_loads_from_project_env(monkeypatch, tmp_path):
    (tmp_path / '.env').write_text(
        'STATUS_WEBHOOK=https://discord.com/api/webhooks/env/token\n',
        encoding='utf-8',
    )
    monkeypatch.delenv('STATUS_WEBHOOK', raising=False)
    monkeypatch.setattr(notifications, 'PROJECT_ROOT', tmp_path)

    assert notifications.load_webhook_url() == (
        'https://discord.com/api/webhooks/env/token'
    )


def test_shell_webhook_overrides_project_env(monkeypatch, tmp_path):
    (tmp_path / '.env').write_text(
        'STATUS_WEBHOOK=https://discord.com/api/webhooks/env/token\n',
        encoding='utf-8',
    )
    monkeypatch.setenv(
        'STATUS_WEBHOOK',
        'https://discord.com/api/webhooks/shell/token',
    )
    monkeypatch.setattr(notifications, 'PROJECT_ROOT', tmp_path)

    assert notifications.load_webhook_url() == (
        'https://discord.com/api/webhooks/shell/token'
    )


def test_embed_respects_discord_limits():
    fields = [(f'Field {index}', 'x' * 2000) for index in range(30)]
    embed = format_embed(
        't' * 400,
        fields=fields,
        description='d' * 5000,
        footer='f' * 3000,
    )

    assert len(embed['title']) == 256
    assert len(embed['description']) == 4096
    assert len(embed['footer']['text']) == 2048
    assert len(embed['fields']) == 25
    assert all(len(field['value']) == 1024 for field in embed['fields'])


def test_training_status_formatting():
    fields = dict(
        (field[0], field[1])
        for field in training_status_fields(
            'pretrain',
            25,
            1_000,
            4.25,
            4.3,
            4.1,
            3e-4,
            100,
            90,
            10_000,
            'snapshot_01.pt',
            '12.50 GB on rank 0',
        )
    )

    assert fields['Current loss'] == '4.2500'
    assert fields['Smoothed loss'] == '4.3000'
    assert fields['Validation loss'] == '4.1000'
    assert fields['ETA'] == '1m 30s'
    assert fields['Peak GPU memory'] == '12.50 GB on rank 0'
    assert format_duration(90_061) == '1d 01h 01m'
    assert format_progress_bar(50, width=10) == '█████░░░░░'
    assert '10.00% complete' in training_status_description(1_000, 10_000)
