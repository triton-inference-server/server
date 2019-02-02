// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include "src/nvrpc/BidirectionalStreamingLifeCycle.h"
#include "src/nvrpc/Interfaces.h"
#include "src/nvrpc/LifeCycleUnary.h"
#include "src/nvrpc/future_std.h"

namespace nvrpc {

template <class LifeCycle, class Resources>
class BaseContext;

template <class Request, class Response, class Resources>
using Context = BaseContext<LifeCycleUnary<Request, Response>, Resources>;

template <class Request, class Response, class Resources>
using StreamingContext =
    BaseContext<BidirectionalStreamingLifeCycle<Request, Response>, Resources>;

template <class LifeCycle, class Resources>
class BaseContext : public LifeCycle {
 public:
  using RequestType = typename LifeCycle::RequestType;
  using ResponseType = typename LifeCycle::ResponseType;
  using ResourcesType = std::shared_ptr<Resources>;
  using QueueFuncType = typename LifeCycle::ExecutorQueueFuncType;
  using LifeCycleType = LifeCycle;

  virtual ~BaseContext() override {}

 protected:
  const ResourcesType& GetResources() const { return m_Resources; }
  double Walltime() const;

  virtual void OnContextStart();
  virtual void OnContextReset();

 private:
  virtual void OnLifeCycleStart() final override;
  virtual void OnLifeCycleReset() final override;

  ResourcesType m_Resources;
  std::chrono::high_resolution_clock::time_point m_StartTime;

  void FactoryInitializer(QueueFuncType, ResourcesType);

  // Factory function allowed to create unique pointers to context objects
  template <class ContextType>
  friend std::unique_ptr<ContextType> ContextFactory(
      typename ContextType::QueueFuncType q_fn,
      typename ContextType::ResourcesType resources);

 public:
  // Convenience method to acquire the Context base pointer from a derived class
  BaseContext<LifeCycle, Resources>* GetBase()
  {
    return dynamic_cast<BaseContext<LifeCycle, Resources>*>(this);
  }
};

// Implementations

/**
 * @brief Method invoked when a request is received and the per-call context
 * lifecycle begins.
 */
template <class LifeCycle, class Resources>
void
BaseContext<LifeCycle, Resources>::OnLifeCycleStart()
{
  m_StartTime = std::chrono::high_resolution_clock::now();
  OnContextStart();
}

template <class LifeCycle, class Resources>
void
BaseContext<LifeCycle, Resources>::OnContextStart()
{
}

/**
 * @brief Method invoked at the end of the per-call lifecycle just before the
 * context is reset.
 */
template <class LifeCycle, class Resources>
void
BaseContext<LifeCycle, Resources>::OnLifeCycleReset()
{
  OnContextReset();
}

template <class LifeCycle, class Resources>
void
BaseContext<LifeCycle, Resources>::OnContextReset()
{
}

/**
 * @brief Number of seconds since the start of the RPC
 */
template <class LifeCycle, class Resources>
double
BaseContext<LifeCycle, Resources>::Walltime() const
{
  return std::chrono::duration<double>(
             std::chrono::high_resolution_clock::now() - m_StartTime)
      .count();
}

/**
 * @brief Used by ContextFactory to initialize the Context
 */
template <class LifeCycle, class Resources>
void
BaseContext<LifeCycle, Resources>::FactoryInitializer(
    QueueFuncType queue_fn, ResourcesType resources)
{
  this->SetQueueFunc(queue_fn);
  m_Resources = resources;
}

/**
 * @brief ContextFactory is the only function in the library allowed to create
 * an IContext object.
 */
template <class ContextType>
std::unique_ptr<ContextType>
ContextFactory(
    typename ContextType::QueueFuncType queue_fn,
    typename ContextType::ResourcesType resources)
{
  auto ctx = nvrpc::make_unique<ContextType>();
  auto base = ctx->GetBase();
  base->FactoryInitializer(queue_fn, resources);
  return ctx;
}

}  // end namespace nvrpc
