
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

/*
template <typename T, typename = typename std::enable_if<std::is_function<T>::value, T>>
class Delegate {
public:
    // Vector to hold the functions
    std::vector<T> functions;

    // Overload += to add a function
    Delegate& operator+=(const T& func) {
        std::cout << &func << std::endl;
        functions.push_back(func);
        return *this;
    }

    // Call all functions
    void operator()(int value) {
        for (const auto& func : functions) {
            func(value);
        }
    }
};
*/
template<typename Signature>
class Delegate;

template<typename Ret, typename... Args>
class Delegate<Ret(Args...)> {
public:
    using FuncType = std::function<Ret(Args...)>;

    struct Inner
    {
        FuncType fn;
        void* adress = nullptr;
    };
    std::vector<Inner> functions;

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
        in.adress = reinterpret_cast<void*>(const_cast<FuncType*>( &func ));
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
};

// Example functions
void MyFunction1(int value) {
    std::cout << "Function 1: " << value << std::endl;
}

void MyFunction2(int value) {
    std::cout << "Function 2: " << value << std::endl;
}

int main() {
    Delegate<void(int)> myDelegate;

    std::function<void(int)> gg1 = MyFunction1;
    std::function<void(int)> gg2 = MyFunction2;
    

    myDelegate += gg2;
    myDelegate += gg1;

    //// Adding functions
    myDelegate += MyFunction1;
    myDelegate += MyFunction2;

    std::function<void(int)> mm = [](int val)
        {
            std::cout << "Function 3: " << val << std::endl;
        };

    myDelegate += mm;

    
    //// Calling all functions
    myDelegate(42);

    //// Removing a function
    myDelegate -= MyFunction1;
    myDelegate -= gg2;

    //// Calling remaining functions
    myDelegate(100);

    //// Comparing delegates
    //Delegate<std::function<void(int)>> anotherDelegate;
    //anotherDelegate += MyFunction2;

    //if (myDelegate == anotherDelegate) {
    //    std::cout << "Delegates are equal!" << std::endl;
    //}
    //else {
    //    std::cout << "Delegates are not equal!" << std::endl;
    //}

    return 0;
}