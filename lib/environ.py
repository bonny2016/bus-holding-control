from dis import dis
from pickle import TRUE
from secrets import token_urlsafe
from tokenize import triple_quoted
from xml.dom import NOT_SUPPORTED_ERR
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np
import data
from collections import deque

PASSENGER_DATA_DIR = "../data/line_2/passenger"
BUS_DATA_DIR = "../data/line_2/bus"
MAX_CAPACITY = 30

# reward per passenger station, simulate ticket income. 
REWARD_PER_PASSENGER_STAION = 1
# reward per stranded passenger
REWARD_PER_STRAND = -2
# reward for each onboarding
REWARD_PER_PASSENGER_ONBOARD = 1
# reward for each passenger waiting minute 
REWARD_PER_PASSENGER_WAITING_MINUTE = -0.2
# reward for each alighting passenger, i.e, complete one riding
REWARD_PER_ALIGHTING = 1
# reward for each bus stopping. simulate bus operational cost
REWARD_PER_BUS_DISPATCH = -30


# maximum steps for an episode.
EPISODE_LENGTH = 1050
# maximum waiting minutes for any passenger. assume passengers leave when waited this long
MAX_WAIT_MINUTES = 15
# reward for each leaving passenger
REWARD_PER_LEFT = -5

# minimum and maximum no. of buses for the pool
MIN_BUSES = 13
MAX_BUSES = 13
# reward for invalid "go" action when no bus is available
REWARD_INVALID_DISPATCH_ACTION = -30

# min and max headway betweeb subsequent dispatch actions.
MIN_HEADWAY = 3
MAX_HEADWAY = 15
#reward for invalid headway 
REWARD_INVALID_HEADWAY = -5

class Actions(enum.Enum):
    HOLD = 0
    DISPATCH = 1

# class for one bus 
class Bus:
    def __init__(self, id, no_stations, max_capacity, current_station=-1) :
        self.id = id
        # total number of stations. 
        self.no_stations = no_stations
        # current_station is the latest passed station 
        self.current_station = current_station
         # passenger_onboard is matrix, passengers_onboard[i,j] is the number of passengers travelling from station i to j
        self.max_capcity = max_capacity
        self.reset()

    def reset(self):
        # passed_minute is the number of minutes this bus has been running since passed current_station
        self.passed_minutes = 0
        # total minutes of current trip if running
        self.trip_minutes = 0
        self.passengers_onboard = np.zeros([self.no_stations, self.no_stations])
        self.trip_id = -1
        self.load = np.array([-1]*self.no_stations)
        self.current_station = -1

    def _on_board(self, m_awaiting_passengers): 
        """
        passengers onboard to the bus at current station to the maximum of it's capacity 
        """
        m_stranded = np.zeros(m_awaiting_passengers.shape)
        # if zero awaiting
        if np.sum(m_awaiting_passengers) == 0:
            return (m_awaiting_passengers, m_stranded)
        # if bus is full already, take zero and leave all as stranded
        elif np.sum(self.passengers_onboard) >= self.max_capcity:
            return (np.zeros(m_awaiting_passengers.shape), m_awaiting_passengers)
        # if bus can take all awaitings without exceeding max capacity
        elif np.sum(self.passengers_onboard) + np.sum(m_awaiting_passengers) <= self.max_capcity:
            self.passengers_onboard[self.current_station] = np.sum(m_awaiting_passengers,1)
            return (m_awaiting_passengers, m_stranded)
        # if bus can take only partial awaitings and leave some stranded
        else:
            m_onboarded = np.zeros(m_awaiting_passengers.shape)
            to_be_onboard = int(self.max_capcity - np.sum(self.passengers_onboard))
            m_stranded = m_awaiting_passengers.copy()
            # for each minute, in reversed order such that passengers waited longest go first
            for i in reversed(range(MAX_WAIT_MINUTES)):
                queue_head_to_all_stops = m_awaiting_passengers[:,i].astype(int)
                if np.sum(queue_head_to_all_stops) > 0:
                    #for each destiny stop
                    for k in range(self.current_station+1, self.no_stations):
                        if queue_head_to_all_stops[k] > 0: 
                            if queue_head_to_all_stops[k] <= to_be_onboard:
                                m_onboarded[k,i] = queue_head_to_all_stops[k]
                                m_stranded[k,i] = 0
                                to_be_onboard -= queue_head_to_all_stops[k]
                            else:
                                m_onboarded[k,i] = to_be_onboard
                                m_stranded[k,i] -= to_be_onboard
                                to_be_onboard = 0
                        if to_be_onboard <=0:
                            break
                if to_be_onboard <=0:
                    break
            self.passengers_onboard[self.current_station] += np.sum(m_onboarded,1)
            return (m_onboarded, m_stranded)           

    def _off_board(self):
        """
        passengers destinied at current station take off
         """
        take_off = np.array(self.passengers_onboard[:,self.current_station])
        self.passengers_onboard[:,self.current_station] = 0
        return take_off

    def _next_stop(self, m3_awaiting_passengers):
        """
        stop at next station, passengers offboard and onboard up to maximum capacity. 
        """
        self.current_station += 1
        self.passed_minutes = 0
        
        # enter waiting pool at the end of jurney
        if self.current_station == self.no_stations:
            self.current_station = -1
            return (np.zeros([self.no_stations]), 
            np.zeros([self.no_stations, MAX_WAIT_MINUTES]), np.zeros([self.no_stations,MAX_WAIT_MINUTES]))
        else:
            list_off_passengers = self._off_board()
            (m_onboarded, m_stranded)  = self._on_board(m3_awaiting_passengers[self.current_station,:,:])
            return (list_off_passengers, m_onboarded, m_stranded)
        
            
    def step(self, list_current_bus_speed, m3_awaiting_passengers):
        """
        move forward for 1 minute, assuming bus is on it's way. (not in the pool)
        :param list_current_bus_speed: 1d array of shape(no_stations,), represents distance between consecutive stations
                in terms of no. of minutes to drive.
        :param m3_awaiting_passengers: 3d array of shape (no_stations, no_stations, MAX_WAIT_MINUTES), 
            where m3_awaiting_passengers(i,j,n) is no. of waiting passengers at station i, destinied to station j, and have waited for n minutes 
        :return a tuple (stopped, list_off_passengers, m_onboarded, m_stranded) where
            1. stopped: boolean represents whether the bus is stopped at this step
            2. list_off_passengers: 1d array of shape(n_stations,) representing no. of offboard passengers onboarded from each station
            3. m_onboarded: 2d array of shape(n_stations,MAX_WAITING_MINUTES) represents no. of passengers onboarded successfully to each station 
            4. m_stranded: 2d array of shape(n_stations,MAX_WAITING_MINUTES) represents no. of passengers of remained waiting in the station due to capacity limitation 
        
        """
        list_off_passengers, m_onboarded, m_stranded = None, None, None
        stopped = False

        if list_current_bus_speed[self.current_station] <= self.passed_minutes or self.current_station == -1:
            (list_off_passengers, m_onboarded, m_stranded) = self._next_stop(m3_awaiting_passengers)
            stopped = True
        
        self.passed_minutes += 1
        self.trip_minutes += 1
        # print("bus.passed_minutes here:",self.passed_minutes)
        return(stopped, list_off_passengers, m_onboarded, m_stranded)
   
# a fleet of buses, including running buses and waiting buses
class BusFleet:
    def __init__(self, no_stations, min_buses = MIN_BUSES, max_buses = MAX_BUSES, max_capacity=MAX_CAPACITY):
        # no. of stations for this bus line
        self.no_stations = no_stations
        # minimum no. of running buses at any given time
        self.min_buses = min_buses
        # maximum no. of running buses at any given time
        self.max_buses = max_buses
        # max capacity for all buses
        self.max_capacity = max_capacity
        self.reset()

    def reset(self):
        self.bus_pool = deque([Bus(id, self.no_stations, self.max_capacity) for id in range(self.min_buses)])
        self.running_buses = deque()

        self.m3_current_awaiting = np.zeros([self.no_stations, self.no_stations, MAX_WAIT_MINUTES])

        self.next_trip_id = 0
        self.trip_loads = deque()
        self.trip_boarding = deque()
        self.trip_alighting = deque()
        self.trip_times = deque()

    def get_running_buses(self):
        n_buses = np.zeros([self.no_stations])
        for b in self.running_buses:
            n_buses[b.current_station] += 1
        return n_buses

    def step(self, list_current_bus_speed, m_new_arrival_passengers, action):
        """
        the fleet proceed for the next minute
        :param list_current_bus_speed: 1d array of shape(no_stations,), represents distance between consecutive stations
            in terms of no. of minutes to drive.
        :param m_new_arrival_passengers: 2d array of shape(no_stations, no_stations), represents no. of new arrivals during this minute. 
            m_new_arrival_passengers[i,j] represents no. new arrival passengers at station i destinied to staion j
        :param action: whether to set off a new bus
        :return a tuple(m_last_off_passengers, m_last_on_passengers, m3_last_stranded_passengers, 
                list_last_bus_stopped, list_current_onboard, list_current_bus_count,
                m3_current_stranded, m3_current_awaiting, m_passenger_left where
        1. m_last_off_passengers: 2d array of shape(no_stations, no_stations).  
           m_last_off_passengers[i,j] reprents no. of passengers offboarded traveling from station i to station j
        2. m_last_on_passengers: 2d array of shape(no_stations, no_stations). 
           m_last_on_passengers[i,j] represented no. of passengers onboarded traveling from station i to station j
        3. m3_last_stranded_passengers: 3d array of shape(no_stations, no_stations, MAX_WAIT_MINUTES)
           m3_last_stranded_passengers[i,j,n] represented no. of stranded passengers traveling from station i to station j and waited n minutes
        4. list_last_bus_stopped: 1d array of shape(no_stations,) represente no. of buses stopped at each station
        5. list_current_onboard: 1d array of shape(no_stations,) represents no. of onboard passengers at each station
            list_current_onboard[i] is the total no. onboard passengers that is travelling on all buses currently between station i and i+1
        6. list_current_bus_count: 1d array of shape(no_stations,) represents no. of running buses at each station
        7. m3_current_stranded: self.m3_current_stranded
        8. m3_current_awaiting: self.m3_current_awaiting
        9. m_passenger_left: 2d array of shape(no_stations, no_stations) represents no. of passengers waited 
           up to MAX_WAIT_MINUTES minutes and no longer wait.

        """
        m_last_off_passengers = np.zeros([self.no_stations, self.no_stations])
        m_last_on_passengers = np.zeros([self.no_stations, self.no_stations])
        m3_last_stranded_passengers = np.zeros([self.no_stations, self.no_stations, MAX_WAIT_MINUTES])
        list_current_onboard = np.zeros(self.no_stations)
        list_last_bus_stopped = np.zeros(self.no_stations) 
        list_current_bus_count = np.zeros(self.no_stations)

        m_last_passenger_left = np.array(self.m3_current_awaiting[:,:,-1]) 
        self.m3_current_awaiting[:,:,1:] = self.m3_current_awaiting[:,:,:-1]
        self.m3_current_awaiting[:,:,0] = m_new_arrival_passengers

        invalid_action = False
        #sef off a new bus if we can
        if action == Actions.DISPATCH:
            new_bus = None
            if len(self.bus_pool) > 0:
                new_bus = self.bus_pool.popleft()
            elif len(self.running_buses) < self.max_buses:
                next_id = len(self.running_buses)
                new_bus = Bus(next_id, self.no_stations, self.max_capacity)
            if new_bus is not None:
                new_bus.trip_id = self.next_trip_id
                self.next_trip_id += 1
                self.running_buses.append(new_bus)
                self.trip_loads.append(np.array([-1]*self.no_stations))
                self.trip_times.append(np.array([-1]*self.no_stations))
                self.trip_boarding.append(np.array([-1]*self.no_stations))
                self.trip_alighting.append(np.array([-1]*self.no_stations))               
            else:
                invalid_action = True

        # all running buses move one step:
        for bus in list(self.running_buses):
            (stopped,  list_off_passengers, m_onboarded, m_stranded)= bus.step(list_current_bus_speed, self.m3_current_awaiting)
            if stopped:
                # if bus.current_station < 2:
                self.trip_loads[bus.trip_id][bus.current_station] = np.sum(bus.passengers_onboard)
                self.trip_alighting[bus.trip_id][bus.current_station] = np.sum(list_off_passengers)
                self.trip_boarding[bus.trip_id][bus.current_station] = np.sum(m_onboarded)
                self.trip_times[bus.trip_id][bus.current_station] = bus.trip_minutes
                m_last_off_passengers[:,bus.current_station] += list_off_passengers
                m_last_on_passengers[bus.current_station] += np.sum(m_onboarded,1)
                m3_last_stranded_passengers[bus.current_station] += m_stranded
                list_last_bus_stopped[bus.current_station] += 1 
                self.m3_current_awaiting[bus.current_station] = self.m3_current_awaiting[bus.current_station] - m_onboarded

                #returing to pool if reached last station
                if bus.current_station == -1:
                    bus.reset()
                    self.running_buses.remove(bus)
                    self.bus_pool.append(bus)
            list_current_onboard[bus.current_station] += np.sum(bus.passengers_onboard)
            list_current_bus_count[bus.current_station] += 1    

        return(m_last_off_passengers, m_last_on_passengers, m3_last_stranded_passengers, m_last_passenger_left,
                list_last_bus_stopped, list_current_onboard, list_current_bus_count,
                self.m3_current_awaiting, invalid_action)
    
# state of multiple bus fleets(routes).   
class State:
    def __init__(self, route_names, total_stations, passenger_data, bus_data, bus_time_ranges, passenger_time_ranges,
                        reward_per_passenger_station = REWARD_PER_PASSENGER_STAION, reward_per_strand = REWARD_PER_STRAND, 
                        reward_per_bus_dispatch = REWARD_PER_BUS_DISPATCH, reward_per_passenger_onboard = REWARD_PER_PASSENGER_ONBOARD,
                        reward_per_passenger_waiting_minute = REWARD_PER_PASSENGER_WAITING_MINUTE,
                        reward_per_passenger_alighting = REWARD_PER_ALIGHTING, reward_per_left_passenger = REWARD_PER_LEFT, state_steps=1):
        # list of route_names
        self.route_names = route_names
        
        # scaler, no. of total stations for all routes
        self.total_stations = total_stations 
        
        # dict {route_name: pair(start_time, end_time)} represent start and end minute of bus operation for each route
        self.bus_time_ranges = bus_time_ranges
        
        # dict{route_name: bus_speed} represent bus speed for each route
        self.bus_data = bus_data
        
        # 3d array of shape(end_time-start_time, station_no, station_no) represents no. of arriving passengers at each minute
        self.passenger_data = passenger_data
        
        self.passenger_time_ranges = passenger_time_ranges
        
        self.bus_routes = {}

        self.reward_per_passenger_station = reward_per_passenger_station
        self.reward_per_strand = reward_per_strand
        self.reward_per_bus_dispatch = reward_per_bus_dispatch
        self.reward_per_passenger_onboard = reward_per_passenger_onboard
        self.reward_per_left_passenger = reward_per_left_passenger
        self.reward_per_passenger_alighting = reward_per_passenger_alighting
        self.reward_per_passenger_waiting_minute = reward_per_passenger_waiting_minute
        self.reward_tables = {}

        # The no. of previous minutes(history) that snapshotted as observation 
        self.state_steps = state_steps

        for route_name in route_names:
            self.bus_routes[route_name] = BusFleet(self.total_stations )
        self.prev_state = None
    
    def _init_reward_table(self, n_stations, reward_per_station):
        """
        simulate reward as bus fare that is propotional to the length of journey. 
        reward[start_station, stop_station] = reward_per_station * (stop_station - start_station)
        """
        reward_table = np.zeros([n_stations, n_stations])
        for i in range(n_stations):
            for j in range(n_stations):
                reward_table[i,j] =  (j - i) * reward_per_station 
        return reward_table

    def reset(self, route_name, offset):
        """
        reset to a particular route, resume to a fresh start: i.e: 
        zero running buses and zero accumulated waiting passengers
        """
        self._route_name  = route_name
        self._passenger_data = self.passenger_data[route_name]
        self._bus_time_range = self.bus_time_ranges[route_name]
        self._passenger_time_range = self.passenger_time_ranges[route_name]
        self._bus_fleet = self.bus_routes[route_name]
        self._bus_fleet.reset()
        self._bus_speed = self.bus_data[route_name]
        self._total_stations = self.total_stations
        self._offset = offset
        self._steps = 0
        self._prev_state = np.zeros(shape=self.shape)
        self._prev_waiting_time = 0
        self._prev_on_passengers = 0
        # record last step stats on each station
        self.list_last_left = np.zeros([self.total_stations])
        self.list_last_boarded = np.zeros([self.total_stations])
        self.list_last_alighted = np.zeros([self.total_stations])
        self.list_last_stranded = np.zeros([self.total_stations])
        self.list_last_waiting_minutes = np.zeros([self.total_stations])
        self.list_last_bus_stopped = np.zeros(self.total_stations)
        
        # record all dispatch starting times
        self.dispatch_times = deque()
        self.trip_states = deque(maxlen=self.state_steps)
        for i in range(self.state_steps):
            self.trip_states.append(-1*np.ones((MAX_BUSES, 3*self.total_stations+1)))

        self.tmp_delta_tot = 0
        self.tmp_m_last_off_passengers = 0
        self.tmp_m_last_on_passengers = 0
        self.tmp_m3_last_stranded_passengers = 0
        self.tmp_m_last_left_passenger = 0
        self.tmp_invalid_action = 0
        self.tmp_outrange_headway = 0
        self.tmp_arrival = 0

    def _calculate_reward(self, action, m_last_off_passengers, m_last_on_passengers, 
                m3_last_stranded_passengers, m_last_left_passengers, invalid_action, 
                outrange_headway, waiting_time_delta):
        total_reward = 0.0

        # collect reward for passengers taken off. simulating ticket income
        total_reward += np.sum(m_last_off_passengers) * self.reward_per_passenger_alighting
        
        # collect reward for onboarded passengers. 
        total_reward += np.sum(m_last_on_passengers) * self.reward_per_passenger_onboard

        # collect penalty for no. of stopped buses, simulating bus running cost
        if action == Actions.DISPATCH and not invalid_action:
            total_reward += self.reward_per_bus_dispatch

        # collect penalty for stranded passengers. 
        total_reward += np.sum(m3_last_stranded_passengers) * self.reward_per_strand

        # collect penalty for left passengers (left for exceeding MAX_WAITING_MINUTES). 
        total_reward += np.sum(m_last_left_passengers) * self.reward_per_left_passenger

        total_reward += (waiting_time_delta)* self.reward_per_passenger_waiting_minute

        if invalid_action:
            total_reward += REWARD_INVALID_DISPATCH_ACTION
        if outrange_headway: 
            # print("outrange_headway:", outrange_headway)
            total_reward += REWARD_INVALID_HEADWAY * outrange_headway
        return total_reward

    def step(self, action):
        """
        Perform one step of action
        :param action
        :return: reward, state, done
        """
        reward = 0.0
        done = False

        if self._offset - self._passenger_time_range[0] < self._passenger_data.shape[0]:
            arrivals = self._passenger_data[self._offset - self._passenger_time_range[0]]
        else:
            arrivals = np.zeros([self._total_stations, self._total_stations])
        # bus move forward for a minute, passengers take on and off 
        (m_last_off_passengers, m_last_on_passengers, m3_last_stranded_passengers, 
        m_last_left_passenger, list_last_bus_stopped, list_current_onboard, list_current_bus_count,
        m3_current_awaiting, invalid_action) = self._bus_fleet.step(
                            list_current_bus_speed = self._bus_speed[self._offset], 
                            m_new_arrival_passengers = arrivals, 
                            action = action)
        self._offset += 1
        self._steps += 1

        self.list_last_left = np.sum(m_last_left_passenger, axis=1)
        self.list_last_boarded = np.sum(m_last_on_passengers, axis=1)
        self.list_last_alighted = np.sum(m_last_off_passengers, axis=0)
        self.list_last_stranded = np.sum(np.sum(m3_last_stranded_passengers,2), axis=1)
        self.list_last_waiting_minutes = np.sum(np.sum(m3_current_awaiting,2), axis=1)
        self.list_last_bus_stopped = list_last_bus_stopped
        last_dispatch_step = self.dispatch_times[-1] if len(self.dispatch_times) > 0 else -1
        headway = self._steps - last_dispatch_step
        outrange_headway = 0
        if action==Actions.DISPATCH and not invalid_action:
            self.dispatch_times.append(self._steps)
            if headway < MIN_HEADWAY and last_dispatch_step != -1:
                outrange_headway = MIN_HEADWAY - headway
            elif headway > MAX_HEADWAY:
                outrange_headway = headway - MAX_HEADWAY
        else:
            if headway > MAX_HEADWAY:
                outrange_headway = headway - MAX_HEADWAY

        waiting_time_delta = np.sum(self.list_last_waiting_minutes)
        reward = self._calculate_reward(action, m_last_off_passengers, m_last_on_passengers, 
                m3_last_stranded_passengers, m_last_left_passenger, invalid_action, 
                outrange_headway, waiting_time_delta)

        self.tmp_delta_tot += waiting_time_delta
        self.tmp_m_last_off_passengers += np.sum(m_last_off_passengers)
        self.tmp_m_last_on_passengers += np.sum(m_last_on_passengers)
        self.tmp_m3_last_stranded_passengers += np.sum(m3_last_stranded_passengers)
        self.tmp_m_last_left_passenger += np.sum(m_last_left_passenger)
        self.tmp_invalid_action += np.sum(invalid_action)
        self.tmp_outrange_headway += np.sum(outrange_headway)
        self.tmp_arrival += np.sum(arrivals)

        done = False
        if self._steps >= EPISODE_LENGTH or self._offset >= self.bus_time_ranges[self._route_name][1] :
            done = True
        
        current_trip_status = self.get_trip_status(running=False)
        self.trip_states.append(current_trip_status)

        return (reward, np.array(self.trip_states).astype(np.float32), done, invalid_action)
    
    def print_state(self):
        print(f"")
    @property
    def shape(self):
        return (self.state_steps, 9, self.total_stations)

    '''
    def encode(self, list_current_onboard = None, list_current_bus_count = None, m3_current_stranded = None, 
                m3_current_awaiting = None, m_last_on_passengers=None, m_last_off_passengers = None, 
                m_last_left_passengers = None):
        result = np.zeros(shape=self.shape)
        if self._prev_state is not None:
            result[:-1] = self._prev_state[1:]
        if list_current_onboard is not None:
            result[-1, 0] = list_current_onboard
        if m3_current_stranded is not None:
            result[-1, 1] = np.sum(np.sum(m3_current_stranded,2),1)
        if m_last_on_passengers is not None:
            result[-1, 6] = np.sum(m_last_on_passengers, 1)
        if m_last_off_passengers is not None:
            result[-1, 7] = np.sum(m_last_off_passengers, 0)
        if m_last_left_passengers is not None:
            result[-1, 8] = np.sum(m_last_left_passengers, 1)
        if m3_current_awaiting is not None: 
            result[-1, 2] = np.sum(np.sum(m3_current_awaiting,2),1)
            cur_waiting_minutes = np.zeros(m3_current_awaiting.shape[:-1])
            for i in range(MAX_WAIT_MINUTES):
                tmp = m3_current_awaiting[:,:,i] * i
                cur_waiting_minutes += tmp
            result[-1, 5, :] = np.sum(cur_waiting_minutes,1)
            # self.accumulated_waiting_minutes += np.sum(np.sum(m3_current_awaiting[:,:,1:],2),1)
            # self.accumulated_waiting_minutes += np.sum(np.sum(m3_current_awaiting[:,:,:],2),1)
            #result[-1, 5, :] = np.mean(total_waiting_minutes,1)
        if (self._offset - self._passenger_time_range[0]) < self._passenger_data.shape[0]:
            result[-1, 3,:] = np.sum(self._passenger_data[self._offset - self._passenger_time_range[0]], 1)
        if list_current_bus_count is not None:
            result[-1, 4,:] = list_current_bus_count
        self._prev_state = result
        return result.astype(np.float32)
    '''

    def get_trip_loads(self, n):
        loads = np.array(self._bus_fleet.trip_loads)
        n = min(n, loads.shape[0])
        return loads[-n:]
    
    def get_trip_boarding(self, n):
        boarding = np.array(self._bus_fleet.trip_boarding)
        n = min(n, boarding.shape[0])
        return boarding[-n:]
    
    def get_trip_alighting(self, n):
        alighting = np.array(self._bus_fleet.trip_alighting)
        n = min(n, alighting.shape[0])
        return alighting[-n:]

    def get_trip_times(self, n):
        times = np.array(self._bus_fleet.trip_times)
        n = min(n, times.shape[0])
        return times[-n:]

    def get_trip_status(self, size=MAX_BUSES, running=False):
        if running:
            n = len(self._bus_fleet.running_buses)
        else:
            n = min(len(self.dispatch_times), size)
        if n == 0:
            # return np.zeros([size,self.total_stations+1]).astype(np.float32)
            return np.concatenate((self._steps * np.ones([size,1]), -1*np.ones([size,3*self.total_stations])),axis=1).astype(np.float32)
            # return -1*np.ones([size,(self.total_stations+1)]).astype(np.float32)

        dispatch_times = np.array(self.dispatch_times)[-n:]
        relative_dispatch_times = self._steps - dispatch_times
        loads = self.get_trip_loads(n)
        boarding = self.get_trip_boarding(n)
        alighting = self.get_trip_alighting(n)
        combined = -1 * np.ones((loads.shape[0], loads.shape[1]*3 + 1))
        for i in range(self.total_stations):
            combined[:,3*i+1]=loads[:,i]
            combined[:,3*i+2]=boarding[:,i]
            combined[:,3*i+3]=alighting[:,i]
        combined[:,0] = relative_dispatch_times
        if combined.shape[0] < size:
            missing_values= np.concatenate((self._steps  * np.ones([size - combined.shape[0],1]), -1*np.ones([size - combined.shape[0],3*self.total_stations])),axis=1)
            combined = np.concatenate((missing_values, combined),axis=0)
        return combined.astype(np.float32)

    def last_step_stats(self):
        list_running_buses = self._bus_fleet.get_running_buses()
        return (self.list_last_boarded, self.list_last_alighted,self.list_last_left,
         self.list_last_stranded, self.list_last_waiting_minutes, list_running_buses) 
    

class BusEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("BusEnv-v0")

    def __init__(self, state_steps, passenger_dir=PASSENGER_DATA_DIR, bus_dir=BUS_DATA_DIR):
        super(BusEnv, self).__init__()
        (route_names, total_stations, passenger_data, passenger_time_ranges,
         bus_data, bus_time_ranges) = data.load_data(passenger_dir, bus_dir)
        print(f"loaded routes:{route_names}")
        self.route_names = route_names

        # assume station no. are same for all routes.
        self.total_stations = total_stations[route_names[0]]
        assert all(value == self.total_stations  for value in total_stations.values())
        self.passenger_data = passenger_data
        self.passenger_time_ranges = passenger_time_ranges
        self.bus_speed = bus_data
        self.bus_time_ranges = bus_time_ranges
        self.bus_data = bus_data
        self.state_steps = state_steps
        self.state =  State(route_names, self.total_stations, passenger_data, bus_data, bus_time_ranges, passenger_time_ranges, state_steps=state_steps)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.state.shape, dtype=np.float32)
        # self.seed()

    def reset(self, pre_steps=0, random=False):
        """
        reset state to fresh start. i.e, no running bus, no awaiting passengers.
        we choose route randomly, and choose current time offset to zero or randomly,
        and we backward current offset further by pre_steps steps. pre_steps is adjusted if reached the minimum time range.

        :param pre_steps: the number of minutes we move backward for staging 
        :param random: bool. we set offset to route start time if false, otherwise set offset randomly between start and end time. 
        :return observation 
        """
        # self._route_name = self.np_random.choice(self.route_names)
        self._route_name = self.route_names[0]
        time_ranges =  self.passenger_time_ranges[self._route_name]
        #time_ranges = [MINUTE_FROM, MINUTE_TO]
        if random:
            offset = self.np_random.choice(
                time_ranges[1] - time_ranges[0]) + time_ranges[0]
        else:
            offset = time_ranges[0]
        #print(self.passenger_data)
        if pre_steps > offset - time_ranges[0]:
            pre_steps = offset - time_ranges[0]
        self.pre_steps = pre_steps

        offset = offset - pre_steps
        
        self.state.reset(self._route_name, offset)
        trip_states = np.array(self.state.trip_states).astype(np.float32)
        return trip_states

    
    def step(self, action_idx):
        action = Actions(action_idx)
        reward, obs, done, invalid_action = self.state.step(action)
        info = {
            "route": self._route_name,
            "invalid_dispatch": invalid_action
        }
        # status = self.state.get_trip_status(running=False)
        # print("obs.shape:", obs.shape)
        return obs, reward, done, info
    

    def get_last_step_stats(self):
        return self.state.last_step_stats()

    def get_trip_loads(self, running=False):
        return self.state.get_trip_loads(running)

    def get_time(self):
        return self.state._offset

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

# verification code
if __name__ ==  "__main__":
    busEnv = BusEnv(1)
    busEnv.reset(pre_steps=0, random=False)
    for i in range(200):
        action_idx = 1 if i%5==0 else 0
        obs, reward, done, info = busEnv.step(action_idx=action_idx )
        if i%10 == 0:
            print(obs[0, 0:3, :])
