import json
import threading

import notifications
from notifications import (
    DiscordNotifier,
    EMBED_COLORS,
    format_embed,
    gpu_telemetry_fields,
    parse_gpu_query,
    parse_performance_reasons,
)
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
    assert fields['Tokens per second'] == '100 TPS'
    assert fields['Peak GPU memory'] == '12.50 GB on rank 0'
    assert format_duration(90_061) == '1d 01h 01m'
    assert format_progress_bar(50, width=10) == '█████░░░░░'
    assert '10.00% complete' in training_status_description(1_000, 10_000)

def test_gpu_query_parser_reports_both_cards():
    output = (
        '0, NVIDIA GeForce RTX 3090, 79, 100, 195.07, 300.00, 1875, 100\n'
        '1, NVIDIA GeForce RTX 3090, 80, 100, 198.72, 300.00, 600, 100\n'
    )

    gpus = parse_gpu_query(output)

    assert [gpu['index'] for gpu in gpus] == ['0', '1']
    assert gpus[0]['temperature'] == '79'
    assert gpus[1]['clock'] == '600'


def test_performance_parser_ignores_counters():
    output = """
    Clocks Event Reasons
        SW Power Cap                 : Not Active
        HW Slowdown
            HW Thermal Slowdown      : Not Active
        SW Thermal Slowdown          : Active
    Clocks Event Reasons Counters
        SW Thermal Slowdown          : 644864106 us
    """

    thermal_active, power_cap_active = parse_performance_reasons(output)

    assert thermal_active
    assert not power_cap_active


def test_gpu_fields_show_every_reported_gpu():
    fields, throttled = gpu_telemetry_fields([
        {
            'index': '0',
            'name': 'NVIDIA GeForce RTX 3090',
            'temperature': '79',
            'fan_speed': '100',
            'power_draw': '195.07',
            'power_limit': '300.00',
            'clock': '1875',
            'utilization': '100',
            'thermal_throttling': False,
            'power_capping': False,
        },
        {
            'index': '1',
            'name': 'NVIDIA GeForce RTX 3090',
            'temperature': '80',
            'fan_speed': '100',
            'power_draw': '198.72',
            'power_limit': '300.00',
            'clock': '600',
            'utilization': '100',
            'thermal_throttling': True,
            'power_capping': False,
        },
    ])

    assert [field[0] for field in fields] == [
        'GPU 0 — NVIDIA GeForce RTX 3090',
        'GPU 1 — NVIDIA GeForce RTX 3090',
    ]
    assert '79°C' in fields[0][1]
    assert '1,875 MHz' in fields[0][1]
    assert 'THERMAL THROTTLING ACTIVE' in fields[1][1]
    assert throttled == ['1']


def test_notifier_adds_both_gpu_stats_on_background_thread(monkeypatch):
    requests = []
    telemetry_threads = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    def fake_collect_gpu_telemetry():
        telemetry_threads.append(threading.current_thread().name)
        return [
            {
                'index': '0',
                'name': 'RTX 3090',
                'temperature': '79',
                'fan_speed': '100',
                'power_draw': '195',
                'power_limit': '300',
                'clock': '1875',
                'utilization': '100',
                'thermal_throttling': False,
                'power_capping': False,
            },
            {
                'index': '1',
                'name': 'RTX 3090',
                'temperature': '80',
                'fan_speed': '100',
                'power_draw': '199',
                'power_limit': '300',
                'clock': '600',
                'utilization': '100',
                'thermal_throttling': True,
                'power_capping': False,
            },
        ], None

    monkeypatch.setattr(notifications, 'urlopen', fake_urlopen)
    monkeypatch.setattr(
        notifications,
        'collect_gpu_telemetry',
        fake_collect_gpu_telemetry,
    )
    monkeypatch.setenv(
        'STATUS_WEBHOOK',
        'https://discord.com/api/webhooks/test/token',
    )

    notifier = DiscordNotifier(make_config())
    notifier.send(
        '📈 TitusAI training progress',
        [('Tokens per second', '22,754 TPS')],
        description='Training normally.',
        color='purple',
        include_gpu_stats=True,
    )

    assert notifier.close()
    assert telemetry_threads == ['titus-discord-notifier']

    payload = json.loads(requests[0][0].data.decode('utf-8'))
    embed = payload['embeds'][0]
    fields = {field['name']: field['value'] for field in embed['fields']}

    assert embed['color'] == EMBED_COLORS['orange']
    assert 'thermal throttling' in embed['title'].lower()
    assert 'GPU 1' in embed['description']
    assert fields['Tokens per second'] == '22,754 TPS'
    assert 'GPU 0 — RTX 3090' in fields
    assert 'GPU 1 — RTX 3090' in fields

