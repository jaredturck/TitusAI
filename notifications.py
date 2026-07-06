import csv
import json
import os
import queue
import socket
import subprocess
import threading
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values

from config import DISCORD_CONFIG, PROJECT_ROOT


EMBED_COLORS = {
    'blue': 0x3498DB,
    'purple': 0x9B59B6,
    'green': 0x2ECC71,
    'orange': 0xF39C12,
    'red': 0xE74C3C,
    'gold': 0xF1C40F,
    'grey': 0x95A5A6,
}


GPU_QUERY_ARGUMENTS = [
    '--query-gpu=index,name,temperature.gpu,fan.speed,power.draw,power.limit,clocks.current.sm,utilization.gpu',
    '--format=csv,noheader,nounits',
]


def run_nvidia_smi(arguments, timeout=3):
    result = subprocess.run(
        ['nvidia-smi', *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def parse_gpu_query(output):
    gpus = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 8:
            continue

        values = [value.strip() for value in row]
        gpus.append({
            'index': values[0],
            'name': values[1],
            'temperature': values[2],
            'fan_speed': values[3],
            'power_draw': values[4],
            'power_limit': values[5],
            'clock': values[6],
            'utilization': values[7],
        })

    return gpus


def parse_performance_reasons(output):
    thermal_active = False
    power_cap_active = False

    for line in output.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith('SW Thermal Slowdown')
            or stripped.startswith('HW Thermal Slowdown')
        ):
            value = stripped.rsplit(':', 1)[-1].strip()
            if value == 'Active':
                thermal_active = True
        elif stripped.startswith('SW Power Cap'):
            value = stripped.rsplit(':', 1)[-1].strip()
            if value == 'Active':
                power_cap_active = True

    return thermal_active, power_cap_active


def collect_gpu_telemetry():
    try:
        gpus = parse_gpu_query(run_nvidia_smi(GPU_QUERY_ARGUMENTS))
        for gpu in gpus:
            performance = run_nvidia_smi([
                '-i',
                gpu['index'],
                '-q',
                '-d',
                'PERFORMANCE',
            ])
            thermal_active, power_cap_active = parse_performance_reasons(
                performance
            )
            gpu['thermal_throttling'] = thermal_active
            gpu['power_capping'] = power_cap_active
        return gpus, None
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        return [], str(error)


def format_metric(value, suffix):
    if value in {'', 'N/A', '[Not Supported]'}:
        return 'Unavailable'
    try:
        number = float(value)
    except ValueError:
        return f'{value}{suffix}'

    if number.is_integer():
        value = f'{int(number):,}'
    else:
        value = f'{number:,.1f}'
    return f'{value}{suffix}'


def gpu_telemetry_fields(gpus, error=None):
    if not gpus:
        return [
            (
                'GPU telemetry',
                clean_text(error or 'No NVIDIA GPUs reported', 1024),
                False,
            )
        ], []

    fields = []
    throttled_gpus = []
    for gpu in gpus:
        if gpu.get('thermal_throttling'):
            throttled_gpus.append(gpu['index'])
            thermal_status = '⚠️ **THERMAL THROTTLING ACTIVE**'
        else:
            thermal_status = '✅ No thermal throttling'

        if gpu.get('power_capping'):
            power_status = ' • Power cap active'
        else:
            power_status = ''

        value = (
            f'Temperature: **{format_metric(gpu["temperature"], "°C")}** • '
            f'Fan: **{format_metric(gpu["fan_speed"], "%")}**\n'
            f'Clock: **{format_metric(gpu["clock"], " MHz")}** • '
            f'Power: **{format_metric(gpu["power_draw"], " W")} / '
            f'{format_metric(gpu["power_limit"], " W")}**\n'
            f'Utilization: **{format_metric(gpu["utilization"], "%")}** • '
            f'{thermal_status}{power_status}'
        )
        fields.append((
            f'GPU {gpu["index"]} — {gpu["name"]}',
            value,
            False,
        ))

    return fields, throttled_gpus


def add_gpu_telemetry(event):
    gpus, error = collect_gpu_telemetry()
    fields, throttled_gpus = gpu_telemetry_fields(gpus, error)
    event['fields'] = list(event.get('fields') or []) + fields

    if throttled_gpus:
        gpu_list = ', '.join(f'GPU {index}' for index in throttled_gpus)
        warning = f'⚠️ **Thermal throttling detected on {gpu_list}.**'
        description = event.get('description')
        event['description'] = (
            f'{warning}\n\n{description}' if description else warning
        )
        event['title'] = f'⚠️ Thermal throttling — {event["title"]}'
        event['color'] = 'orange'

    return event


def load_webhook_url():
    environment_url = os.environ.get('STATUS_WEBHOOK', '').strip()
    if environment_url:
        return environment_url

    values = dotenv_values(PROJECT_ROOT / '.env')
    return str(values.get('STATUS_WEBHOOK') or '').strip()


def clean_text(value, maximum_length):
    text = str(value)
    if len(text) <= maximum_length:
        return text
    return text[:maximum_length - 1] + '…'


def format_embed(title, fields=None, description=None, color='blue', footer=None):
    embed = {
        'title': clean_text(title, 256),
        'color': EMBED_COLORS.get(color, EMBED_COLORS['blue']),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'footer': {
            'text': clean_text(
                footer or f'TitusAI Training Monitor • {socket.gethostname()}',
                2048,
            ),
        },
    }

    if description:
        embed['description'] = clean_text(description, 4096)

    if fields:
        embed['fields'] = []
        for field in fields[:25]:
            if len(field) == 2:
                name, value = field
                inline = True
            else:
                name, value, inline = field

            embed['fields'].append({
                'name': clean_text(name, 256),
                'value': clean_text(value, 1024),
                'inline': bool(inline),
            })

    return embed


class DiscordNotifier:
    def __init__(self, config=None):
        self.config = DISCORD_CONFIG if config is None else config
        self.webhook_url = load_webhook_url()
        self.enabled = bool(self.config['enabled'] and self.webhook_url)
        self.messages = queue.Queue()
        self.worker = None
        self.last_error = None

        if self.config['enabled'] and not self.webhook_url:
            print(
                '[!] Discord notifications are enabled but STATUS_WEBHOOK '
                'is missing from .env'
            )

        if self.enabled:
            self.worker = threading.Thread(
                target=self._run,
                name='titus-discord-notifier',
                daemon=True,
            )
            self.worker.start()

    def send(
        self,
        title,
        fields=None,
        body=None,
        color='blue',
        description=None,
        footer=None,
        include_gpu_stats=False,
    ):
        if not self.enabled:
            return

        event = {
            'title': title,
            'fields': fields,
            'description': description if description is not None else body,
            'color': color,
            'footer': footer,
            'include_gpu_stats': include_gpu_stats,
        }
        self.messages.put(event)

    def _run(self):
        while True:
            event = self.messages.get()
            if event is None:
                self.messages.task_done()
                return

            include_gpu_stats = event.pop('include_gpu_stats', False)
            if include_gpu_stats:
                event = add_gpu_telemetry(event)
            embed = format_embed(**event)
            self._post(embed)
            self.messages.task_done()

    def _post(self, embed):
        payload = {
            'username': self.config['username'],
            'embeds': [embed],
            'allowed_mentions': {'parse': []},
        }
        request = Request(
            self.webhook_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'TitusAI/1.0',
            },
            method='POST',
        )

        try:
            with urlopen(
                request,
                timeout=self.config['request_timeout_seconds'],
            ) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            self.last_error = str(error)
            print(f'[!] Discord notification failed: {error}')

    def flush(self):
        if not self.enabled or self.worker is None:
            return False

        self.messages.join()
        return self.last_error is None and self.worker.is_alive()

    def close(self):
        if not self.enabled or self.worker is None:
            return self.last_error is None

        self.messages.put(None)
        self.worker.join(
            timeout=self.config['request_timeout_seconds'] + 1
        )
        return self.last_error is None and not self.worker.is_alive()


def main():
    notifier = DiscordNotifier()
    if not notifier.enabled:
        print('[!] Add STATUS_WEBHOOK to .env and try again')
        return 1

    notifier.send(
        '✅ TitusAI notifications configured',
        [
            ('Host', socket.gethostname()),
            ('Delivery', 'Discord embed'),
            ('Status', 'Webhook test successful'),
        ],
        description='Remote training updates are ready.',
        color='green',
        include_gpu_stats=True,
    )

    if notifier.close():
        print('[+] Discord test notification sent')
        return 0

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
