import check_setup
import prepare_data
import process_utils


class FlushTracker:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def flush(self):
        self.events.append(self.name)


def test_hard_exit_flushes_output_before_exit(monkeypatch):
    events = []
    monkeypatch.setattr(process_utils.sys, 'stdout', FlushTracker(events, 'stdout'))
    monkeypatch.setattr(process_utils.sys, 'stderr', FlushTracker(events, 'stderr'))
    monkeypatch.setattr(process_utils.os, '_exit', lambda code: events.append(('exit', code)))

    process_utils.hard_exit_after_success()

    assert events == ['stdout', 'stderr', ('exit', 0)]


def test_check_setup_exits_only_after_main_succeeds(monkeypatch):
    events = []
    monkeypatch.setattr(check_setup, 'main', lambda: events.append('main'))
    monkeypatch.setattr(
        check_setup,
        'hard_exit_after_success',
        lambda: events.append('exit'),
    )

    check_setup.run()

    assert events == ['main', 'exit']


def test_prepare_data_exits_only_after_main_succeeds(monkeypatch):
    events = []
    monkeypatch.setattr(prepare_data, 'main', lambda: events.append('main'))
    monkeypatch.setattr(
        prepare_data,
        'hard_exit_after_success',
        lambda: events.append('exit'),
    )

    prepare_data.run()

    assert events == ['main', 'exit']


def test_failed_main_does_not_hard_exit(monkeypatch):
    events = []

    def fail():
        raise RuntimeError('example failure')

    monkeypatch.setattr(check_setup, 'main', fail)
    monkeypatch.setattr(
        check_setup,
        'hard_exit_after_success',
        lambda: events.append('exit'),
    )

    try:
        check_setup.run()
    except RuntimeError:
        pass

    assert events == []
