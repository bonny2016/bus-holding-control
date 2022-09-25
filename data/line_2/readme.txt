We use data collected from BRT (Bus Rapid Transit) Line 2 of Xiamen city, China on a random day in June 2018 as our experiment data. BRT Line 2 is a high-freqency bus line system which is a good candidate for real-time control. The environment is simulated based on real passenger swiping records and bus travel time information, with some pre-processing steps to simulate passenger arrival times based on corresponding boarding time, since passenger arrival times are not directly recorded in swiping records. A uniform distribution of arrivals during two consecutive buses at each stop is assumed.

The file "passenger\direction_0.csv" is the pre-processed passengers data based on bus swipping record, where each row represent a passenger's information:

    1. ID - unique ID. 
    2. BOARD_MINUTE - the actual boarding time recorded by swipping record represented as # of minute
    3. FROM_STATION - the origin stop# of this passenger 
    4. TO_STATION - the destiny stop# of this passenger 
    5. ARRIVAL_MINUTE - the arrival time of this passenger represented as # of minute.

The file "bus\direction_0.csv" file is the bus riding data based on bus cruising record of a particular day. The header of this file is "time_h1,time_h2,time_m1,time_m2,FROM_MINUTE,TO_MINUTE,s0,s1, ... ,s36".  Each row represent the bus cruising minutes between consecutive bus stops during a specific time period (FROM_MINUTE to TO_MINUTE), where "0" means that there is NO buses running during that peroid of time.

The file "bus\direction_0_dispatch.csv" represent the actual dispatch time from stop 0.






