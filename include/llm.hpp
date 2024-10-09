//
//  llm.hpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//

#ifndef LLM_hpp
#define LLM_hpp

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <functional>
#include <unordered_map>

#include "ortwrapper.hpp"
#include "tokenizer.hpp"
#include "json.hpp"

using namespace Ort;
using json = nlohmann::json;
class Tokenizer;
class Pipeline;
class LlmConfig;

// Llm start
// llm stream buffer with callback
class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;;
    LlmStreamBuffer(CallBack callback) : callback_(callback) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, n);
        }
        return n;
    }

private:
    CallBack callback_ = nullptr;
};

enum PROMPT_TYPE {
    SYSTEM = 0,
    ATTACHMENT = 1,
    USER = 2,
    ASSISTANT = 3,
    OTHER = 4
};

struct Prompt {
    PROMPT_TYPE type;
    std::string str;
    std::vector<int> tokens;
};

class Llm {
public:
    using PromptItem = std::pair<std::string, std::string>; // <role, content>
    Llm(std::shared_ptr<LlmConfig> config) : config_(config) {}
    virtual ~Llm();
    void chat();
    void reset();
    static Llm* createLLM(const std::string& config_path);
    virtual void load();
    Value forward(const std::vector<int>& input_ids);
    int sample(Value& logits, const std::vector<int>& pre_ids);
    std::string apply_prompt_template(const std::string& user_content) const;
    std::string apply_chat_template(const std::vector<PromptItem>& chat_prompts) const;
    std::string response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr);
    std::string response(const std::vector<PromptItem>& chat_prompts, std::ostream* os = &std::cout, const char* end_with = nullptr);
    void generate_init();
    std::string generate(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    void print_speed();
    // config function
    std::string dump_config();
    bool set_config(const std::string& content);
    friend class Pipeline;
public:
    // forward info
    int prompt_len_ = 0;
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    std::vector<int> history_ids_;
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
    bool is_single_ = true;
    bool attention_fused_ = true;
protected:
    std::shared_ptr<LlmConfig> config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::vector<int> key_value_shape_ = {};
    Value past_key_values_ {nullptr};
    std::shared_ptr<RuntimeManager> runtime_manager_;
    std::shared_ptr<Module> module_;
    void init_runtime();
    std::string decode(int id);
    bool is_stop(int token_id);
    virtual std::vector<int> tokenizer(const std::string& query);
    virtual Value embedding(const std::vector<int>& input_ids);
    virtual Value gen_attention_mask(int seq_len);
    virtual Value gen_position_ids(int seq_len);
};
// Llm end

#endif // LLM_hpp
