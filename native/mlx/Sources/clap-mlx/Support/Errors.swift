import Foundation

enum WorkerError: Error, CustomStringConvertible {
  case invalidModelDirectory(String)

  var description: String {
    switch self {
    case .invalidModelDirectory(let message): message
    }
  }
}
