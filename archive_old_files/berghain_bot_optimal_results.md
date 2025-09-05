
❯ python berghain_bot_optimal.py

============================================================
🎯 TESTING OPTIMAL BOT - SCENARIO 1
============================================================
2025-09-05 18:00:48,672 - INFO - 🎯 Started game 5f6d6981
2025-09-05 18:00:48,673 - INFO - 📋 Constraints: [('young', 600), ('well_dressed', 600)]
2025-09-05 18:00:49,087 - INFO - 👥 Person 0: Admitted 1, Rejected 0 | Progress: young: 0.2%, well_dressed: 0.0%
2025-09-05 18:01:32,361 - INFO - 👥 Person 200: Admitted 102, Rejected 99 | Progress: young: 11.2%, well_dressed: 10.8%
2025-09-05 18:02:15,769 - INFO - 👥 Person 400: Admitted 201, Rejected 200 | Progress: young: 19.5%, well_dressed: 22.5%
2025-09-05 18:02:58,702 - INFO - 👥 Person 600: Admitted 304, Rejected 297 | Progress: young: 30.3%, well_dressed: 33.7%
2025-09-05 18:03:41,895 - INFO - 👥 Person 800: Admitted 397, Rejected 404 | Progress: young: 40.2%, well_dressed: 41.8%
2025-09-05 18:04:25,444 - INFO - 👥 Person 1000: Admitted 497, Rejected 504 | Progress: young: 49.8%, well_dressed: 52.2%
2025-09-05 18:05:08,579 - INFO - 👥 Person 1200: Admitted 596, Rejected 605 | Progress: young: 61.3%, well_dressed: 61.5%
2025-09-05 18:05:51,382 - INFO - 👥 Person 1400: Admitted 712, Rejected 689 | Progress: young: 72.2%, well_dressed: 73.8%
2025-09-05 18:06:34,666 - INFO - 👥 Person 1600: Admitted 831, Rejected 770 | Progress: young: 82.5%, well_dressed: 84.5%
2025-09-05 18:07:17,935 - INFO - 👥 Person 1800: Admitted 929, Rejected 872 | Progress: young: 91.2%, well_dressed: 93.8%
2025-09-05 18:07:43,682 - ERROR - ❌ FAILED: Venue full but constraints not met: young: 585/600 (need 15 more), well_dressed: 598/600 (need 2 more). Rejected 920 people
2025-09-05 18:07:43,682 - INFO - 📊 young: 585/600 ❌
2025-09-05 18:07:43,682 - INFO - 📊 well_dressed: 598/600 ❌

📈 RESULT:
Status: failed
Rejections: 920

============================================================
🎯 TESTING OPTIMAL BOT - SCENARIO 2
============================================================
2025-09-05 18:07:43,900 - INFO - 🎯 Started game 708a511a
2025-09-05 18:07:43,900 - INFO - 📋 Constraints: [('techno_lover', 650), ('well_connected', 450), ('creative', 300), ('berlin_local', 750)]
2025-09-05 18:07:44,309 - INFO - 👥 Person 0: Admitted 1, Rejected 0 | Progress: techno_lover: 0.2%, well_connected: 0.0%, creative: 0.0%, berlin_local: 0.0%
2025-09-05 18:08:27,031 - INFO - 👥 Person 200: Admitted 192, Rejected 9 | Progress: techno_lover: 19.2%, well_connected: 18.4%, creative: 4.0%, berlin_local: 9.9%
2025-09-05 18:09:10,204 - INFO - 👥 Person 400: Admitted 384, Rejected 17 | Progress: techno_lover: 40.3%, well_connected: 35.6%, creative: 9.3%, berlin_local: 18.0%
2025-09-05 18:09:53,259 - INFO - 👥 Person 600: Admitted 574, Rejected 27 | Progress: techno_lover: 59.1%, well_connected: 55.6%, creative: 15.0%, berlin_local: 28.7%
2025-09-05 18:10:36,369 - INFO - 👥 Person 800: Admitted 767, Rejected 34 | Progress: techno_lover: 79.5%, well_connected: 77.1%, creative: 19.0%, berlin_local: 38.0%
2025-09-05 18:11:19,636 - INFO - 👥 Person 1000: Admitted 944, Rejected 57 | Progress: techno_lover: 95.1%, well_connected: 99.1%, creative: 22.0%, berlin_local: 49.7%
2025-09-05 18:11:32,923 - ERROR - ❌ FAILED: Venue full but constraints not met: creative: 69/300 (need 231 more), berlin_local: 397/750 (need 353 more). Rejected 63 people
2025-09-05 18:11:32,923 - INFO - 📊 techno_lover: 650/650 ✅
2025-09-05 18:11:32,923 - INFO - 📊 well_connected: 475/450 ✅
2025-09-05 18:11:32,923 - INFO - 📊 creative: 69/300 ❌
2025-09-05 18:11:32,923 - INFO - 📊 berlin_local: 397/750 ❌

📈 RESULT:
Status: failed
Rejections: 63

============================================================
🎯 TESTING OPTIMAL BOT - SCENARIO 3
============================================================
2025-09-05 18:11:33,190 - INFO - 🎯 Started game 02c2a5e3
2025-09-05 18:11:33,190 - INFO - 📋 Constraints: [('underground_veteran', 500), ('international', 650), ('fashion_forward', 550), ('queer_friendly', 250), ('vinyl_collector', 200), ('german_speaker', 800)]
2025-09-05 18:11:33,629 - INFO - 👥 Person 0: Admitted 1, Rejected 0 | Progress: underground_veteran: 0.2%, international: 0.0%, fashion_forward: 0.0%, queer_friendly: 0.0%, vinyl_collector: 0.0%, german_speaker: 0.1%
2025-09-05 18:12:16,803 - INFO - 👥 Person 200: Admitted 195, Rejected 6 | Progress: underground_veteran: 28.2%, international: 18.2%, fashion_forward: 25.8%, queer_friendly: 4.4%, vinyl_collector: 5.0%, german_speaker: 11.1%
2025-09-05 18:12:59,872 - INFO - 👥 Person 400: Admitted 386, Rejected 15 | Progress: underground_veteran: 54.2%, international: 34.8%, fashion_forward: 49.8%, queer_friendly: 7.2%, vinyl_collector: 9.5%, german_speaker: 22.5%
2025-09-05 18:13:42,954 - INFO - 👥 Person 600: Admitted 579, Rejected 22 | Progress: underground_veteran: 80.0%, international: 53.5%, fashion_forward: 73.1%, queer_friendly: 10.8%, vinyl_collector: 14.5%, german_speaker: 33.2%
2025-09-05 18:14:26,205 - INFO - 👥 Person 800: Admitted 771, Rejected 30 | Progress: underground_veteran: 105.0%, international: 70.5%, fashion_forward: 97.1%, queer_friendly: 12.8%, vinyl_collector: 17.0%, german_speaker: 44.6%
2025-09-05 18:15:09,455 - INFO - 👥 Person 1000: Admitted 959, Rejected 42 | Progress: underground_veteran: 132.2%, international: 87.8%, fashion_forward: 120.9%, queer_friendly: 17.6%, vinyl_collector: 20.0%, german_speaker: 55.9%
2025-09-05 18:15:18,573 - ERROR - ❌ FAILED: Venue full but constraints not met: international: 587/650 (need 63 more), queer_friendly: 47/250 (need 203 more), vinyl_collector: 44/200 (need 156 more), german_speaker: 473/800 (need 327 more). Rejected 42 people
2025-09-05 18:15:18,573 - INFO - 📊 underground_veteran: 693/500 ✅
2025-09-05 18:15:18,573 - INFO - 📊 international: 587/650 ❌
2025-09-05 18:15:18,573 - INFO - 📊 fashion_forward: 692/550 ✅
2025-09-05 18:15:18,573 - INFO - 📊 queer_friendly: 47/250 ❌
2025-09-05 18:15:18,573 - INFO - 📊 vinyl_collector: 44/200 ❌
2025-09-05 18:15:18,574 - INFO - 📊 german_speaker: 473/800 ❌

📈 RESULT:
Status: failed
Rejections: 42
