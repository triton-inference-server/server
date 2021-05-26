#ifndef DCGM_DCGM_API_EXPORT_H
#define DCGM_DCGM_API_EXPORT_H

#undef DCGM_PUBLIC_API
#undef DCGM_PRIVATE_API
​
#if defined(DCGM_API_EXPORT)
#define DCGM_PUBLIC_API __attribute((visibility("default")))
#else
#define DCGM_PUBLIC_API
#if defined(ERROR_IF_NOT_PUBLIC)
#error(Should be public)
#endif
#endif
​
#define DCGM_PRIVATE_API __attribute((visibility("hidden")))​

#endif // DCGM_DCGM_API_EXPORT_H