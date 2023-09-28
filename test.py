import os

# what's provided by package
print(dir(os))

cpus = os.cpu_count()
print(cpus)

# The number of usable CPUs can be obtained with
# - only available on some Unix platforms.
# usable_cpus = len(os.sched_getaffinity(0))
# print(usable_cpus)
