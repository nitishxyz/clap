import Foundation

struct QueuedChat {
  let id: String?
  let control: ControlRequest
  let data: Data
  let receivedNs: UInt64
}

enum WorkerSchedulingEvent {
  case started(String?)
  case pendingCancelled(String?)
  case queueChanged(Int)
}
