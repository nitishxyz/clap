#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>

namespace {

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char ch : value) {
    if (ch == '\'') out += "'\\''";
    else out += ch;
  }
  out += "'";
  return out;
}

std::string extract_string(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return "";
  pos = json.find('"', pos + 1);
  if (pos == std::string::npos) return "";

  std::string value;
  bool escape = false;
  for (std::size_t index = pos + 1; index < json.size(); ++index) {
    const char ch = json[index];
    if (escape) {
      switch (ch) {
        case 'n': value += '\n'; break;
        case 'r': value += '\r'; break;
        case 't': value += '\t'; break;
        default: value += ch; break;
      }
      escape = false;
    } else if (ch == '\\') {
      escape = true;
    } else if (ch == '"') {
      return value;
    } else {
      value += ch;
    }
  }
  return value;
}

double extract_double(const std::string& json, const std::string& key, double fallback) {
  const std::string needle = "\"" + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return fallback;
  try {
    return std::stod(json.substr(pos + 1));
  } catch (...) {
    return fallback;
  }
}

int extract_int(const std::string& json, const std::string& key, int fallback) {
  const std::string needle = "\"" + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return fallback;
  try {
    return std::stoi(json.substr(pos + 1));
  } catch (...) {
    return fallback;
  }
}

std::string prompt_from_messages(const std::string& json) {
  std::ostringstream prompt;
  std::size_t pos = 0;
  while ((pos = json.find("\"content\"", pos)) != std::string::npos) {
    const auto colon = json.find(':', pos + 9);
    const auto quote = json.find('"', colon == std::string::npos ? pos + 9 : colon);
    if (quote == std::string::npos) break;
    std::string content;
    bool escape = false;
    for (std::size_t index = quote + 1; index < json.size(); ++index) {
      const char ch = json[index];
      if (escape) {
        switch (ch) {
          case 'n': content += '\n'; break;
          case 'r': content += '\r'; break;
          case 't': content += '\t'; break;
          default: content += ch; break;
        }
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        pos = index + 1;
        break;
      } else {
        content += ch;
      }
    }
    if (!content.empty()) prompt << content << "\n";
  }
  return prompt.str();
}

std::filesystem::path executable_dir(const char* argv0) {
  std::error_code ec;
  auto path = std::filesystem::weakly_canonical(argv0, ec);
  if (ec) path = std::filesystem::absolute(argv0, ec);
  if (ec) return std::filesystem::current_path();
  return path.parent_path();
}

std::filesystem::path llama_cli_path(const char* argv0) {
  if (const char* configured = std::getenv("CLAP_LLAMA_CLI")) {
    if (configured[0] != '\0') return configured;
  }
  return executable_dir(argv0) / "llama-cli";
}

std::string build_command(const std::filesystem::path& llama_cli, const std::string& model, const std::string& prompt, int max_tokens, double temperature) {
  std::ostringstream command;
  command
    << shell_quote(llama_cli.string())
    << " --model " << shell_quote(model)
    << " --prompt " << shell_quote(prompt)
    << " --n-predict " << max_tokens
    << " --temp " << temperature
    << " --no-display-prompt";
  return command.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string request;
    if (!std::getline(std::cin, request) || request.empty()) {
      std::cout << "{\"error\":\"expected one JSON request line on stdin\"}\n";
      return 2;
    }

    const std::string model = extract_string(request, "model");
    if (model.empty()) {
      std::cout << "{\"error\":\"request.model is required\"}\n";
      return 2;
    }

    std::string prompt = prompt_from_messages(request);
    if (prompt.empty()) prompt = extract_string(request, "prompt");
    if (prompt.empty()) prompt = "Hello";

    const int max_tokens = extract_int(request, "max_tokens", 256);
    const double temperature = extract_double(request, "temperature", 0.7);
    const auto llama_cli = llama_cli_path(argc > 0 ? argv[0] : "clap-llama");
    if (!std::filesystem::exists(llama_cli)) {
      std::cout << "{\"error\":\"llama-cli not found; build with bun run runtime:llama:build or set CLAP_LLAMA_CLI\"}\n";
      return 127;
    }

    const std::string command = build_command(llama_cli, model, prompt, max_tokens, temperature);

    std::array<char, 4096> buffer{};
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
      std::cout << "{\"error\":\"failed to launch llama-cli\"}\n";
      return 127;
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
      std::cout << "{\"token\":\"" << json_escape(buffer.data()) << "\"}\n";
      std::cout.flush();
    }

    const int status = pclose(pipe);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::cout << "{\"error\":\"llama-cli exited non-zero; see llama worker stderr log\"}\n";
      return 1;
    }
    std::cout << "{\"done\":true}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cout << "{\"error\":\"" << json_escape(error.what()) << "\"}\n";
    return 1;
  }
}
