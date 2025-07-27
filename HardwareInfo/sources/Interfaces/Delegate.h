#pragma once

#include <vector>
#include <functional>
#include <algorithm>

template<typename Signature>
class Delegate;

template<typename Ret, typename... Args>
class Delegate<Ret(Args...)> {
public:
    using FuncType = std::function<Ret(Args...)>;

    Delegate& operator=(Ret(*set)(Args...)) {
        functions.clear();
        this->operator+=(std::move(set));
        return *this;
    }

    Delegate& operator=(const FuncType& func) {
        functions.clear();
        this->operator+=(std::move(func));
        return *this;
    }

    Delegate& operator+=(Ret(*set)(Args...)) {
        functions.emplace_back(Inner(FuncType(set), set));
        return *this;
    }

    // Overload += to add a function
    Delegate& operator+=(const FuncType& func) {
        Inner in;
        in.adress = reinterpret_cast<void*>(const_cast<FuncType*>(&func));
        in.fn = func;
        functions.emplace_back(in);
        return *this;
    }

    Delegate& operator-=(Ret(*set)(Args...)) {
        functions.erase(std::remove_if(functions.begin(), functions.end(),
            [&set](const Inner& f) {
                return  f.adress == (void*)set;
            }), functions.end());
        return *this;
    }

    Delegate& operator-=(const FuncType& func) {
        functions.erase(std::remove_if(functions.begin(), functions.end(),
            [&func](const Inner& f) {
                return  f.adress == reinterpret_cast<void*>(const_cast<FuncType*>(&func));
            }), functions.end());
        return *this;
    }

    void operator()(Args... args) {
        for (const auto& func : functions) {
            func.fn(args...);
        }
    }

    // Collect return values
    std::vector<Ret> InvokeAll(Args... args) const {
        std::vector<Ret> results;
        for (const auto& func : functions) {
            results.push_back(func.fn(args...));
        }
        return results;
    }

    void Reset()
    {
        functions.clear();
    }

private:
    struct Inner
    {
        FuncType fn;
        void* adress = nullptr;
    };
    std::vector<Inner> functions;
};
