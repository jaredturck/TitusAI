import json

import notifications
from notifications import DiscordNotifier, format_discord_message
from train import format_duration, training_status_fields


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


def test_discord_notifier_posts_message(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(notifications, 'urlopen', fake_urlopen)
    monkeypatch.setenv('STATUS_WEBHOOK', 'https://discord.com/api/webhooks/test/token')

    notifier = DiscordNotifier(make_config())
    notifier.send('Training started', [('Step', '1'), ('Loss', '5.0000')])

    assert notifier.close()
    assert len(requests) == 1

    request, timeout = requests[0]
    payload = json.loads(request.data.decode('utf-8'))
    assert timeout == 1
    assert payload['username'] == 'TitusAI Test'
    assert '**Training started**' in payload['content']
    assert 'Step: 1' in payload['content']
    assert payload['allowed_mentions'] == {'parse': []}


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


def test_discord_message_stays_within_limit():
    message = format_discord_message('Status', body='x' * 3000)
    assert len(message) == 2000


def test_training_status_formatting():
    fields = dict(training_status_fields(
        'pretrain',
        25,
        1_000,
        4.25,
        4.1,
        3e-4,
        100,
        90,
        10_000,
        'snapshot_01.pt',
    ))

    assert fields['Progress'] == '10.00%'
    assert fields['Loss'] == '4.2500'
    assert fields['Validation'] == '4.1000'
    assert fields['ETA'] == '1m 30s'
    assert format_duration(90_061) == '1d 01h 01m'
