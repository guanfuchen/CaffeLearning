/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 * solver工厂允许注册solvers，和layer工厂类似。运行时期间，注册求解器能被通过传递SolverParameter protobuffer到CreateSolver
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 * 有两种方法来注册一个求解器solver，template <typename Dtype>以及对应实现
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * 对应的类型为去除Solver的名字，比如MyAwesomeSolver去除Solver的type为MyAwesome
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * 如果求解器通过下述简单的构造期简单创建，需要增加如下REGISTER_SOLVER_CLASS(MyAwesome);
 * If the solver is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  // 求解器注册器类型为map<string, Creator>
  typedef std::map<string, Creator> CreatorRegistry;

  // 静态求解器注册器，Solver求解器
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  // 增加一个创建器，关于Solver求解器，主要有SGD，Adam等等
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry(); // 函数调用，返回静态实例
    // 如果register count求解器类型type为0，也就是已经注册该求解器，那么输出FATAL关于该求解器Solver已经被注册
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    // 然后将SolverRegister map中的相应type设置为该creator
    registry[type] = creator;
  }

  // Get a solver using a SolverParameter.
  // 使用SolverParameter获取solver
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
    // param求解器类型，比如SGD, Adam
    const string& type = param.type();
    // 注册创建器
    CreatorRegistry& registry = Registry();
    // 未知的求解器类型
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    return registry[type](param);
  }

  // Solver类型列表
  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry(); // std::map<string, Creator>
    vector<string> solver_types;
    // registry注册器
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      // solver_types压入iter CreatorRegistry的第一个参数，也就是map，比如SGD等等
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  // Solver registry永远都不能实例化-所有的函数都通过静态变量完成
  SolverRegistry() {}

  // 返回求解器类型列表String
  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    // 实际调用solver_types迭代输出相应的string类型的solver types
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    // 输出的solver type列表应该是"SGD, Adam, ..."
    return solver_types_str;
  }
};


// 求解器注册器
template <typename Dtype>
class SolverRegisterer {
 public:
  // 求解器注册器初始化函数
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    // LOG(INFO) << "Registering solver type: " << type;
    // 调用SolverRegistry中的AddCreator增加type类型的creator，比如SGD的creator
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};

// 定义求解器solver creator，求解器注册器g_creator_f_SGD和g_creator_d_SGD，也就是同时包括float和double的求解器
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

// 定义求解器solver注册类，比如Creator_SGDSolver，这里面包含创建template Creator_XXX，同时调用REGISTER_SOLVER_CREATOR，将该Solver返回
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
