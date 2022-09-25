there are 6 column names.


Historical records

No_num : Unique identification label of the passenger	
Get_in_time_th : the card swiping time of the passenger(the boarding time of the passenger）
Get_in_station：the boarding station of the passenger	
Get_off_station：the alighting station of the passenger	


Simulation data

arrical_station_time：time of passenger arrival at the station 

For the passengers boarding on the same bus at the same station, we assume that their arrival time is
uniformly distributed. Thus, the arrival time of each passenger can be estimated via the boarding time
and the time when the previous bus left the station, and is added in the record.  


time_th means ： How many minutes have passed since 00:00 AM(e.g. 6:21AM = 381st minute)


the other information of lines is mentioned in our paper : Deep Reinforcement Learning based Dynamic Optimization of  Bus Timetable



