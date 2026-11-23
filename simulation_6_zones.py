import numpy as np

num_patches_per_epoch = 1000 #* 848

zone_names = [
            "single_fiber_wm",
            "crossing_fiber_wm",
            "cortical_boundary",
            "subcortical_gm",
            "brainstem_cerebellum",
            "csf_boundary"
        ]

single_fiber_wm = 0
crossing_fiber_wm = 0
cortical_boundary = 0
subcortical_gm = 0
brainstem_cerebellum = 0
csf_boundary = 0

for n in range(num_patches_per_epoch):
    zone_name = zone_names[int(np.random.randint(0, len(zone_names)))]
    if zone_name == "single_fiber_wm":
        single_fiber_wm += 1
    elif zone_name == "crossing_fiber_wm":
        crossing_fiber_wm += 1
    elif zone_name == "cortical_boundary":
        cortical_boundary += 1
    elif zone_name == "subcortical_gm":
        subcortical_gm += 1
    elif zone_name == "brainstem_cerebellum":
        brainstem_cerebellum += 1
    elif zone_name == "csf_boundary":
        csf_boundary += 1

print("single_fiber_wm [%]: ", "%.1f" % (100 * single_fiber_wm / num_patches_per_epoch))
print("crossing_fiber_wm [%]: ", "%.1f" % (100 * crossing_fiber_wm / num_patches_per_epoch))
print("cortical_boundary [%]: ", "%.1f" % (100 * cortical_boundary / num_patches_per_epoch))
print("subcortical_gm [%]: ", "%.1f" % (100 * subcortical_gm / num_patches_per_epoch))
print("brainstem_cerebellum [%]: ", "%.1f" % (100 * brainstem_cerebellum / num_patches_per_epoch))
print("csf_boundary [%]: ", "%.1f" % (100 * csf_boundary / num_patches_per_epoch))

# RESULTS for num_patches_per_epoch = 1000
# single_fiber_wm [%]:  17.3
# crossing_fiber_wm [%]:  17.1
# cortical_boundary [%]:  16.0
# subcortical_gm [%]:  18.8
# brainstem_cerebellum [%]:  15.6
# csf_boundary [%]:  15.2


# RESULTS for num_patches_per_epoch = 1000 * 848
# single_fiber_wm [%]:  16.6
# crossing_fiber_wm [%]:  16.6
# cortical_boundary [%]:  16.6
# subcortical_gm [%]:  16.6
# brainstem_cerebellum [%]:  16.7
# csf_boundary [%]:  16.7