import json
import os
import queue
import socket
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values

from config import DISCORD_CONFIG, PROJECT_ROOT


def load_webhook_url():
    environment_url = os.environ.get('STATUS_WEBHOOK', '').strip()
    if environment_url:
        return environment_url

    values = dotenv_values(PROJECT_ROOT / '.env')
    return str(values.get('STATUS_WEBHOOK') or '').strip()


def format_discord_message(title, fields=None, body=None):
    lines = [f'**{title}**']

    if fields:
        lines.append('```text')
        for label, value in fields:
            lines.append(f'{label}: {value}')
        lines.append('```')

    if body:
        lines.append(str(body))

    return '\n'.join(lines)[:2000]


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

    def send(self, title, fields=None, body=None):
        if not self.enabled:
            return

        message = format_discord_message(title, fields, body)
        self.messages.put(message)

    def _run(self):
        while True:
            message = self.messages.get()
            if message is None:
                self.messages.task_done()
                return

            self._post(message)
            self.messages.task_done()

    def _post(self, message):
        payload = {
            'content': message,
            'username': self.config['username'],
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
        'TitusAI Discord notifications configured',
        [
            ('Host', socket.gethostname()),
            ('Status', 'Webhook test successful'),
        ],
    )

    if notifier.close():
        print('[+] Discord test notification sent')
        return 0

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
