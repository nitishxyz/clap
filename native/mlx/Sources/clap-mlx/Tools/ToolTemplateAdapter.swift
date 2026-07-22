import Foundation
import MLXLMCommon

struct ToolsEnvelope: Decodable {
  let tools: [JSONValue]?
}
