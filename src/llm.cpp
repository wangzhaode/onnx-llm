//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <regex>

#include "llm.hpp"
#include "llmconfig.hpp"
#include "tokenizer.hpp"

#ifdef LLM_SUPPORT_VISION
#include "httplib.h"
#endif

// Llm start
std::string Llm::dump_config() {
    return config_->config_.dump();
}

bool Llm::set_config(const std::string& content) {
    config_->config_.merge_patch(content.c_str());
    return true;
}

void Llm::init_runtime() {
    runtime_manager_.reset(new RuntimeManager());
}

void Llm::load() {
    init_runtime();
    // init module status
    key_value_shape_ = config_->key_value_shape();
    is_single_ = config_->is_single();
    attention_fused_ = config_->attention_fused();
    {
        std::ifstream embedding_bin(config_->embedding_file());
        embedding_bin.close();
    }
    // 1. load vocab
    printf("load tokenizer\n");
    tokenizer_.reset(Tokenizer::createTokenizer(config_->tokenizer_file()));
    printf("load tokenizer Done\n");
    // 3. load model
    int layer_nums = config_->layer_nums();
    key_value_shape_.insert(key_value_shape_.begin(), layer_nums);
    std::string model_path = config_->llm_model();
    printf("load %s ... ", model_path.c_str());
    module_.reset(new Module(runtime_manager_, model_path));
    printf("Load Module Done!\n");
}

Value Llm::forward(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    std::vector<Value> inputs;
    inputs.emplace_back(embedding(input_ids));
    inputs.emplace_back(gen_attention_mask(seq_len));
    inputs.emplace_back(gen_position_ids(seq_len));
    inputs.emplace_back(std::move(past_key_values_));
    auto outputs = module_->onForward(inputs);
    auto logits = std::move(outputs[0]);
    past_key_values_ = std::move(outputs[1]);
    all_seq_len_ += seq_len;
    gen_seq_len_++;
    return logits;
}

int Llm::sample(Value& logits, const std::vector<int>& pre_ids) {
    std::unordered_set<int> ids_set(pre_ids.begin(), pre_ids.end());
    auto scores = logits.GetTensorMutableData<float>();
    auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    // repetition penalty
    const float repetition_penalty = 1.1;
    for (auto id : ids_set) {
        float score = scores[id];
        scores[id] = score < 0 ? score * repetition_penalty : score / repetition_penalty;
    }
    // argmax
    float max_score = 0;
    int token_id = 0;
    for (int i = 0; i < size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id = i;
        }
    }
    return token_id;
}

static std::string apply_template(std::string prompt_template, const std::string& content, const std::string& role = "") {
    if (prompt_template.empty()) return content;
    if (!role.empty()) {
        const std::string placeholder = "%r";
        size_t start_pos = prompt_template.find(placeholder);
        if (start_pos == std::string::npos) return content;
        prompt_template.replace(start_pos, placeholder.length(), role);
    }
    const std::string placeholder = "%s";
    size_t start_pos = prompt_template.find(placeholder);
    if (start_pos == std::string::npos) return content;
    prompt_template.replace(start_pos, placeholder.length(), content);
    return prompt_template;
}

std::string Llm::apply_prompt_template(const std::string& user_content) const {
    auto chat_prompt = config_->prompt_template();
    return apply_template(chat_prompt, user_content);
}

std::string Llm::apply_chat_template(const std::vector<PromptItem>& chat_prompts) const {
    auto chat_template = config_->chat_template();
    std::string prompt_result;
    auto iter = chat_prompts.begin();
    for (; iter != chat_prompts.end() - 1; ++iter) {
        prompt_result += apply_template(chat_template, iter->second, iter->first);
    }
    if (iter->first == "user") {
        prompt_result += apply_prompt_template(iter->second);
    } else {
        prompt_result += apply_template(chat_template, iter->second, iter->first);
    }
    return prompt_result;
}

void Llm::chat() {
    std::vector<PromptItem> history;
    history.push_back(std::make_pair("system", "You are a helpful assistant."));
    while (true) {
        std::cout << "\nQ: ";
        std::string user_str;
        std::cin >> user_str;
        if (user_str == "/exit") {
            break;
        }
        if (user_str == "/reset") {
            history.resize(1);
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        std::cout << "\nA: " << std::flush;
        history.emplace_back(std::make_pair("user", user_str));
        auto assistant_str = response(history);
        history.emplace_back(std::make_pair("assistant", assistant_str));
        std::cout << std::endl;
    }
}

void Llm::reset() {
    history_ids_.clear();
    all_seq_len_ = 0;
}

void Llm::generate_init() {
    // init status
    gen_seq_len_ = 0;
    prefill_us_ = 0;
    decode_us_ = 0;
    past_key_values_ = _Input<float>(key_value_shape_, runtime_manager_);
    if (!config_->reuse_kv()) {
        all_seq_len_ = 0;
        history_ids_.clear();
    }
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_new_tokens) {
    generate_init();
    std::vector<int> output_ids, all_ids = input_ids;
    prompt_len_ = static_cast<int>(input_ids.size());
    if (max_new_tokens < 0) { max_new_tokens = config_->max_new_tokens(); }
    // prefill
    auto logits = forward(input_ids);
    int token = sample(logits, all_ids);
    output_ids.push_back(token);
    all_ids.push_back(token);
    // decode
    while (gen_seq_len_ < max_new_tokens) {
        logits = forward({token});
        token = sample(logits, all_ids);
        if (is_stop(token)) { break; }
        output_ids.push_back(token);
        all_ids.push_back(token);
    }
    return output_ids;
}

std::string Llm::generate(const std::vector<int>& input_ids, std::ostream* os, const char* end_with) {
    prompt_len_ = static_cast<int>(input_ids.size());
    history_ids_.insert(history_ids_.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    auto st = std::chrono::system_clock::now();
    auto logits = forward(input_ids);
    int token = sample(logits, history_ids_);
    auto et = std::chrono::system_clock::now();
    std::string output_str = decode(token);
    prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    *os << output_str << std::flush;
    while (gen_seq_len_ < config_->max_new_tokens()) {
        st = std::chrono::system_clock::now();
        history_ids_.push_back(token);
        logits = forward({token});
        token = sample(logits, history_ids_);
        et = std::chrono::system_clock::now();
        decode_us_ += std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if (is_stop(token)) {
            *os << end_with << std::flush;
            break;
        }
        auto word = decode(token);
        *os << word << std::flush;
        output_str += word;
    }
#ifdef DUMP_PROFILE_INFO
    print_speed();
#endif
    return output_str;
}

std::vector<int> Llm::tokenizer(const std::string& query) {
    auto prompt = apply_prompt_template(query);
    auto input_ids = tokenizer_->encode(prompt);
    return input_ids;
}

std::string Llm::response(const std::string& user_content, std::ostream* os, const char* end_with) {
    generate_init();
    if (!end_with) { end_with = "\n"; }
    std::vector<int> input_ids;
    if (config_->reuse_kv()) {
        auto prompt = apply_prompt_template(user_content);
        if (all_seq_len_ > 0) {
            prompt = "<|im_end|>\n" + prompt;
        }
        input_ids = tokenizer_->encode(prompt);
    } else {
        input_ids = tokenizer(user_content);
    }
    return generate(input_ids, os, end_with);
}

std::string Llm::response(const std::vector<PromptItem>& chat_prompts, std::ostream* os, const char* end_with) {
    if (chat_prompts.empty()) { return ""; }
    generate_init();
    if (!end_with) { end_with = "\n"; }
    auto prompt = apply_chat_template(chat_prompts);
    if (config_->reuse_kv() && all_seq_len_ > 0) {
        prompt = "<|im_end|>\n" + prompt;
    }
    // std::cout << "# prompt : " << prompt << std::endl;
    auto input_ids = tokenizer_->encode(prompt);
    // printf("input_ids (%lu): ", input_ids.size()); for (auto id : input_ids) printf("%d, ", id); printf("\n");
    return generate(input_ids, os, end_with);
}

void Llm::print_speed() {
    auto prefill_s = prefill_us_ * 1e-6;
    auto decode_s = decode_us_ * 1e-6;
    auto total_s = prefill_s + decode_s;
    printf("\n#################################\n");
    printf(" total tokens num  = %d\n", prompt_len_ + gen_seq_len_);
    printf("prompt tokens num  = %d\n", prompt_len_);
    printf("output tokens num  = %d\n", gen_seq_len_);
    printf("  total time = %.2f s\n", total_s);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("  total speed = %.2f tok/s\n", (prompt_len_ + gen_seq_len_) / total_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len_ / prefill_s);
    printf(" decode speed = %.2f tok/s\n", gen_seq_len_ / decode_s);
    printf("   chat speed = %.2f tok/s\n", gen_seq_len_ / total_s);
    printf("##################################\n");
}


Llm::~Llm() {
    module_.reset();
    runtime_manager_.reset();
}

Value Llm::embedding(const std::vector<int>& input_ids) {
    // disk embedding to save memory
    int hidden_size = config_->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());
    auto inputs_embeds = _Input<float>({seq_len, 1, hidden_size}, runtime_manager_);
    size_t size = hidden_size * sizeof(int16_t);
    FILE* file = fopen(config_->embedding_file().c_str(), "rb");
    std::unique_ptr<int16_t[]> buffer(new int16_t[hidden_size]);
    for (size_t i = 0; i < seq_len; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        size_t bytes_read = fread(buffer.get(), 1, size, file);
        (void)bytes_read;
        auto ptr = inputs_embeds.GetTensorMutableData<int16_t>() + i * hidden_size * 2;
        for (int j = 0; j < hidden_size; j++) {
            ptr[j * 2] = 0;
            ptr[j * 2 + 1] = buffer[j];
        }
    }
    fclose(file);
    return std::move(inputs_embeds);
}

std::string Llm::decode(int id) {
    std::string word = tokenizer_->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length()-1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word = static_cast<char>(num);
    }
    return word;
}

Value Llm::gen_attention_mask(int seq_len) {
    int kv_seq_len = all_seq_len_ + seq_len;
    if (seq_len == 1) {
        kv_seq_len = seq_len;
    }
    if (config_->attention_mask() == "float") {
        auto attention_mask = _Input<float>({1, 1, seq_len, kv_seq_len}, runtime_manager_);
        auto ptr = attention_mask.GetTensorMutableData<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                int row = i + all_seq_len_;
                ptr[kv_seq_len * i + j] = (j > row) * std::numeric_limits<float>::lowest();
            }
        }
        return attention_mask;
    } else {
        auto attention_mask = _Input<int>({1, 1, seq_len, kv_seq_len}, runtime_manager_);
        auto ptr = attention_mask.GetTensorMutableData<int>();
        if (config_->attention_mask() == "glm") {
            // chatglm
            for (int i = 0; i < seq_len * kv_seq_len; i++) {
                ptr[i] = 0;
            }
            if (seq_len > 1) {
                for (int i = 1; i < seq_len; i++) {
                    ptr[seq_len * i - 1] = 1;
                }
            }
        } else {
            bool is_glm2 = config_->attention_mask() == "glm2";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    int row = i + all_seq_len_;
                    ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
                }
            }
        }
        return attention_mask;
    }
}

Value Llm::gen_position_ids(int seq_len) {
    if (config_->attention_mask() == "glm") {
        // chatglm
        auto position_ids = _Input<int>({1, 2, seq_len}, runtime_manager_);
        auto ptr = position_ids.GetTensorMutableData<int>();
        if (seq_len == 1) {
            ptr[0] = all_seq_len_ - gen_seq_len_ - 2;
            ptr[1] = gen_seq_len_ + 1;
        } else {
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i] = i;
                ptr[seq_len + i] = 0;
            }
            ptr[seq_len - 1] = seq_len - 2;
            ptr[2 * seq_len - 1] = 1;
        }
        return position_ids;
    } else {
        bool is_glm2 = config_->attention_mask() == "glm2";
        auto position_ids = _Input<int>({1, seq_len}, runtime_manager_);
        auto ptr = position_ids.GetTensorMutableData<int>();
        if (seq_len == 1) {
            ptr[0] = is_glm2 ? gen_seq_len_ : all_seq_len_;
        } else {
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + all_seq_len_;
            }
        }
        return position_ids;
    }
}

bool Llm::is_stop(int token_id) {
    return tokenizer_->is_stop(token_id);
}

Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual()) {
        // llm = new Lvlm(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}