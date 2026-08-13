"""Control signals must reach the recorder even when the frame queue is full.

Regression guard for the hang where a blocking put() of FINALIZE/stop signals on a
full, non-draining queue wedged the per-scene lock and left the dataset unfinalized.
Runs standalone (`python test_recorder_control_delivery.py`) or under pytest.
"""
import queue
import threading

from guide_core.scene.scene_recorder import SceneRecorder


def test_control_signal_delivered_when_queue_full():
    rec = SceneRecorder("pkg", "task", {})  # no thread started, no lerobot needed
    # Saturate with droppable frames so the queue is completely full.
    for _ in range(rec.record_queue.maxsize):
        rec.record_queue.put_nowait({"frame": True})
    assert rec.record_queue.full()

    # Deliver a control signal from another thread; it must NOT block (old code did).
    done = threading.Event()
    threading.Thread(target=lambda: (rec.put_record_data("FINALIZE"), done.set())).start()
    assert done.wait(2.0), "put_record_data('FINALIZE') blocked on a full queue"

    # The signal is in the queue and a frame was evicted to make room.
    items = [rec.record_queue.get_nowait() for _ in range(rec.record_queue.qsize())]
    assert "FINALIZE" in items
    assert rec._dropped_frames >= 1


if __name__ == "__main__":
    test_control_signal_delivered_when_queue_full()
    print("ok")
