import Foundation
import ClapMLXModel
import MLXLMCommon

func promptMessages(_ messages: [ChatMessage]) -> [PromptMessage] {
  messages.map { message in
    PromptMessage(role: message.role, content: message.content,
      toolCalls: message.tool_calls?.map {
        PromptToolCall(name: $0.function.name, arguments: $0.function.arguments)
      })
  }
}

func promptTokenizerAdapter(_ tokenizer: any MLXLMCommon.Tokenizer) -> PromptTokenizerAdapter {
  PromptTokenizerAdapter(eosTokenId: tokenizer.eosTokenId,
    encode: { text, addSpecialTokens in
      tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }, applyChatTemplate: { messages, tools, additionalContext in
      try tokenizer.applyChatTemplate(messages: messages, tools: tools,
        additionalContext: additionalContext ?? [:])
    })
}
