import os
import csv
import glob
import numpy as np
import collections

MAX_STATIONS = 100
LATEST_MINUTE = 25*60 # lastest buses may be running in midnight. for convinience.
PASSENGER_DATA_DIR = "../data/line_2/passenger"
BUS_DATA_DIR = "../data/line_2/bus"


def read_passenger_csv(file_name):
    """
    read daily passenger arrival data from a csv file.  
    :param file_name: csv file name including the following three columns: 
                      FROM_STATION,TO_STATION,ARRIVAL_MINUTE
    :return: a numpy array 'm_result' of shape [total_minutes, total_stations, total_stations]
             where m_result[i,j,k] represent the number of new passengers 
             that arrived at minute i at station j and heading to station k.
    """
    print(f"Reading passenger data file {file_name}...")
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=',')
        h = next(reader)
        indices = [h.index(s) for s in ('FROM_STATION', 'TO_STATION', 'ARRIVAL_MINUTE')]
        
        max_station_id = 0
        
        min_minute = LATEST_MINUTE
        max_minute = 0  

        # initiate with 24 hours data with MAX_STATIONS stations at beginning and trim later
        m_arrival = np.zeros([24*60, MAX_STATIONS, MAX_STATIONS])
        counter = 0
        for row in reader:
            from_station, to_station, arrival_minute = list(map(int, [row[idx] for idx in indices]))
            if from_station > max_station_id:
                max_station_id = from_station
            if to_station > max_station_id:
                max_station_id = to_station
            if arrival_minute < min_minute:
                min_minute = arrival_minute
            if arrival_minute > max_minute:
                max_minute = arrival_minute
            m_arrival[arrival_minute][from_station][to_station] += 1
            counter += 1    
    print(f"""Read done, got {counter} passengers at station 0-{max_station_id} 
            during minute {min_minute}-{max_minute}""")
    m_result = m_arrival[min_minute:max_minute + 1, 0:max_station_id + 1, 0:max_station_id + 1]
    return (m_result, min_minute, max_minute)

def load_passenger_data_from_dir(basedir):
    passengers = {}
    time_ranges = {}
    for path in glob.glob(os.path.join(basedir, "*.csv")):
        file_name = os.path.basename(path).split(".")[0]
        m_result, min_minute, max_minute = read_passenger_csv(path)
        passengers[file_name] = m_result
        time_ranges[file_name] = (min_minute, max_minute)
    return (passengers, time_ranges)

def read_bus_data_csv(file_name, no_stations):
    """
    read daily bus speed data from a csv file.
    :param file_name: csv file name including the following columns 
                      "FROM_MINUTE,TO_MINUTE,s0,s1,...s{no_stations-1}"
                      representing estimated travelling minutes between consequtive stops 
                      for a bus during period [FROM_MINUTE, TO_MINUTE) 
                                          
    :return: a numpy array 'm_result' of shape [total_minutes,no_stations]
             where m_result[minute_no, stop_no] represent estimated travelling time in minutes   
             for a bus at time [minute_no] from stop [stop_no] to the next [stop_no+1]. 
    """
    print(f"Reading bus data file {file_name}...")
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=',')
        h = next(reader)
        stop_list = [f's{idx}' for idx in range(no_stations)]
        indices = [h.index(s) for s in (['FROM_MINUTE', 'TO_MINUTE'] + (stop_list))]
        
        # initiate with 25 hours data with zeros (easy for calculation, buses runs to midnight 1:00am sometimes)
        total_minutes = LATEST_MINUTE
        m_result = np.zeros([total_minutes,no_stations])
        start_minute = -1
        end_minute = 0
        for row in reader:
            from_minute, to_minute, *arrival_minutes = list(map(int, [row[idx] for idx in indices]))
            if(sum(arrival_minutes) > 0):
                if (start_minute <= 0):
                    start_minute = from_minute
                end_minute = to_minute
            m_result[from_minute:to_minute,:] = arrival_minutes
    print("Read done.")
    #print(f"result shape:{m_result.shape}")
    return (m_result, start_minute, end_minute)

def load_bus_data_from_dir(basedir, route_names, total_stations):
    result, time_ranges = {}, {}
    for path in glob.glob(os.path.join(basedir, "*.csv")):
        file_name = os.path.basename(path).split(".")[0]
        if file_name in route_names:
            no_stations = total_stations[file_name]
            (m_result, start_minute, end_minute) = read_bus_data_csv(path, no_stations)
            result[file_name] = m_result
            time_ranges[file_name] = (start_minute, end_minute)
    return result, time_ranges


def load_data(passenger_dir = PASSENGER_DATA_DIR, bus_dir=BUS_DATA_DIR):
    """
    read passenger and bus data from provided directories.
    :param passenger_dir: directory contains passenger csv files.
    :param bus_dir: directory contains bus csv files.
    
    :return: a tuple (route_names, total_stations, passenger_data, bus_data)
             where route_names is a list of route names;
             total_stations is a dictionary containing total number of stops for each route;
             passenger_data is a dictionary containing passenger data for each route;
             bus_data is a dictionary containing bus data for each route.
    """
    route_names = []
    total_stations = {}
    (passenger_data, time_ranges)= load_passenger_data_from_dir(passenger_dir)
    print(f"Finished loading passenger data from {len(passenger_data)} file(s)")
    print(f"time_ranges:{time_ranges}, passenger_data:")
    #print(passenger_data)
    for route_name in passenger_data:
        route_names.append(route_name)
        total_stations[route_name] = passenger_data[route_name].shape[1]
        print("passenger_data.shape:",passenger_data[route_name].shape)
        tmp = np.sum(passenger_data[route_name],0)
        print("now shape:", tmp.shape)
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                if (tmp[i,j]>0 and j<=i):
                    print(f"tmp{i}{j}={tmp[i,j]}")
    
    bus_data, bus_time_ranges = load_bus_data_from_dir(bus_dir,route_names,total_stations)
    print(f"Finished loading bus data from {len(bus_data)} file(s)")
    return route_names, total_stations, passenger_data, time_ranges, bus_data, bus_time_ranges

if __name__  == "__main__":
    route_names, total_stations, passenger_data, passenger_time_ranges, bus_data, bus_time_ranges = load_data()
    print(f"Loaded data with {len(route_names)} routes: {', '.join(route_names)}")
    print(f"bus time ranges: {bus_time_ranges}")
    print(f"passenger time ranges: {passenger_time_ranges}")
    