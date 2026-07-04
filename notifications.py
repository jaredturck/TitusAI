import json
import os
import queue
import socket
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

    def send(self, title, fields=None, body=None, color='blue', description=None, footer=None):
        if not self.enabled:
            return

        event = {
            'title': title,
            'fields': fields,
            'description': description if description is not None else body,
            'color': color,
            'footer': footer,
        }
        self.messages.put(event)

    def _run(self):
        while True:
            event = self.messages.get()
            if event is None:
                self.messages.task_done()
                return

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
    )

    if notifier.close():
        print('[+] Discord test notification sent')
        return 0

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
