import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.integrate import cumtrapz

 
# Fix random seed
np.random.seed(seed=9999)



def run_cycle(dt, price, capped_lot, parking_lot_size, plot_flag):
    
    if np.mod(12/dt,1) != 0:
        raiseException('Non-integer intervals.')
    
    intervals = int(12/dt + 1)
    # Number of intervals in the simulation
    # = 5 implies evaluated every 12/(5-1) = 3 hours
    # = 9 implies evaluated every 12/(9-1) = 1.5 hours
    # = 13 implies evaluated every 12/(13-1) = 1.0 hours
    
    # Simulation time, in hours
    time = np.linspace(0,12,intervals)
    
    # The first element is zero, so simply remove it
    time = time[1:]
    
    # Define some functions to generate the random number of cars. Note that if the
    # time-step changes, then the inputs change as well, as the rates were defined
    # on a 3-hour basis. Note that if dt is too small, underflow issues occur with
    # the random integer generator.
    
    # function to define random input generator
    def rand_cars_in(t):
        
        output = np.zeros_like(t)
        
        for n in range(0,len(t)):
            if 0 < t[n] and t[n] <= 3:
                output[n] = np.rint(np.random.uniform(low=50*dt/3, high=150*dt/3))
            elif 3 < t[n] and t[n] <= 6:
                output[n] = np.rint(np.random.uniform(low=25*dt/3, high=75*dt/3))
            elif 6 < t[n] and t[n] <= 9:
                output[n] = np.rint(np.random.uniform(low=40*dt/3, high=60*dt/3))
            else:
                output[n] = np.rint(np.random.uniform(low=10*dt/3, high=30*dt/3))
                
        return output
    
    # function to define random output generator
    def rand_cars_out(t):
        
        output = np.zeros_like(t)
        
        for n in range(0,len(t)):
            if 0 < t[n] and t[n] <= 3:
                output[n] = np.rint(np.random.uniform(low=5*dt/3, high=15*dt/3))
            elif 3 < t[n] and t[n] <= 6:
                output[n] = np.rint(np.random.uniform(low=10*dt/3, high=30*dt/3))
            elif 6 < t[n] and t[n] <= 9:
                output[n] = np.rint(np.random.uniform(low=40*dt/3, high=60*dt/3))
            else:
                output[n] = np.rint(np.random.uniform(low=50*dt/3, high=150*dt/3))
                
        return output
    
    # Evaluate the number of cars that tried to come in and the number that left
    cars_in = rand_cars_in(time)
    cars_out = rand_cars_out(time)
    net_influx = cars_in - cars_out
    
    # Array for cars parked
    parked_cars = np.zeros_like(time)
    # Total number of cars serviced
    cars_serviced = np.zeros_like(time)
    
    # Assume that all cars leave and come simultaneously at the beginning of the
    # time step. Each value of the time array is treated as the end of a step.
    # Incoming cars will take the spots of leaving cars during the time
    # interval; however, if there are not enough spaces, then the cars will not be
    # able to stay and will leave with the lot full.
    
    # The array, total_cars, represents the number of cars at the end of the step.
    # Because the number of cars is always assumed to be zero at the beginning of
    # the day, it is implicitly assumed that the first value of total_cars simply
    # represents the number of cars at the end of the first interval.
    for n in range(0,len(time)):
    
        # Calculate net change in total number of cars in parking lot.
        if n == 0:
            parked_cars[n] = net_influx[n]
        else:
            parked_cars[n] = parked_cars[n-1] + net_influx[n]
        
        # The total number of cars cannot be less than zero.
        if parked_cars[n] < 0:
            parked_cars[n] = 0
        
        # Is the parking_lot_size capped?
        if capped_lot:
    
            # If the lot is capped, then the number of cars that are serviced is
            # the input cars less the overage amount.
            if parked_cars[n]>parking_lot_size:
                
                if n == 0:
                    cars_serviced[n] = cars_in[n] - (parked_cars[n]-parking_lot_size)
                else:
                    cars_serviced[n] = cars_in[n] - (parked_cars[n]-parking_lot_size) + cars_serviced[n-1]
                # Additionally, force the number of parked cars back to lot size.
                parked_cars[n] = parking_lot_size
            
            else:
                if n == 0:
                    cars_serviced[n] = cars_in[n] 
                else:
                    cars_serviced[n] = cars_in[n] + cars_serviced[n-1]
    
        else:
    
            if n == 0:
                cars_serviced[n] = cars_in[n]
            else:
                cars_serviced[n] = cars_in[n] + cars_serviced[n-1]
    
    daily_income = cars_serviced[-1]*price
    
    # Plotting for diagnostic testing
    if plot_flag:
        # Prepend zero cars at the beginning of the day.
        time = np.insert(time,0,0)
        parked_cars = np.insert(parked_cars,0,0)
        cars_serviced = np.insert(cars_serviced,0,0)
        
        plt.figure(1)
        plt.plot(time,parked_cars,'b-')
        plt.plot(time,cars_serviced,'r-')
        plt.grid()
        plt.xlabel('Time (hours)')
        plt.ylabel('Parked Cars')
        plt.legend(('Parked Cars','Total Cars Serviced'),loc='upper right')

    return daily_income, parked_cars

def run_batch(dt, price, capped_lot, parking_lot_size, cycles, plot_flag):
    
    income = np.zeros(cycles)
    max_cars = np.zeros(cycles)
    for i in range(0,cycles):
        daily_income, parked_cars = run_cycle(dt, price, capped_lot, parking_lot_size, plot_flag)
        income[i] = daily_income
        max_cars[i] = max(parked_cars)

    mu_i, std_i = norm.fit(income)
    x_i = np.linspace(min(income),max(income),1001)
    p_i = norm.pdf(x_i, mu_i, std_i)
    
    try:
        kde_i = gaussian_kde(income)
    except:
        kde_i = None
        print('kde_i not calculated, likely single valued.')

    mu_c, std_c = norm.fit(max_cars)
    x_c = np.linspace(min(max_cars),max(max_cars),1001)
    p_c = norm.pdf(x_c, mu_c, std_c)
    
    try:
        kde_c = gaussian_kde(max_cars)
    except:
        kde_c = None
        print('kde_c not calculated, likely single valued.')

    
    return income, max_cars, mu_i, std_i, x_i, p_i, kde_i, mu_c, std_c, x_c, p_c, kde_c



"""

   Test 1: Run a batch with default parameters

"""

# Hyper parameters

# Time differential in hours
dt = 3

# Income per car
price = 10

# Is the size of the parking lot capped?
capped_lot = False
# If capped, what is the parking lot size?
parking_lot_size = 8

# Number of days in cycle
cycles = 1000

# Run batch
income, max_cars, mu_i, std_i, x_i, p_i, kde_i, mu_c, std_c, x_c, p_c, kde_c = run_batch(dt, price, capped_lot, parking_lot_size, cycles, True)

# Plots

plt.figure(2)
plt.hist(income, edgecolor='k')
plt.xlabel('Daily Income')
plt.ylabel('Count')
plt.savefig('Figures\Income_Counts_dt_3.tiff', dpi=300)

plt.figure(3)
plt.hist(income, density=True, edgecolor='k', label='Data')
plt.plot(x_i, kde_i(x_i), 'r--')
plt.plot(x_i, p_i, 'k-')
plt.xlabel('Daily Income')
plt.ylabel('Probability Density')
plt.legend(('Data','KDE','Norm PDF'))
plt.savefig('Figures\Income_PDF_dt_3.tiff', dpi=300)

plt.figure(4)
plt.hist(max_cars, edgecolor='k')
plt.xlabel('Peak Car Total')
plt.ylabel('Count')
plt.savefig('Figures\Max_Cars_Counts_dt_3.tiff', dpi=300)

plt.figure(5)
plt.hist(max_cars, density=True, edgecolor='k', label='Data')
plt.plot(x_c,kde_c(x_c),'r--')
plt.plot(x_c, p_c, 'k', linewidth=2)
plt.xlabel('Peak Car Total')
plt.ylabel('Probability Density')
plt.legend(('Data','KDE','Norm PDF'))
plt.savefig('Figures\Max_Cars_PDF_dt_3.tiff', dpi=300)

"""

   Test 2: Run a batch with smaller time interval

"""


# Hyper parameters

# Time differential in hours
dt = 1

# Income per car
price = 10

# Is the size of the parking lot capped?
capped_lot = False
# If capped, what is the parking lot size?
parking_lot_size = 8

# Number of days in cycle
cycles = 1000

# Run batch
income, max_cars, mu_i, std_i, x_i, p_i, kde_i, mu_c, std_c, x_c, p_c, kde_c = run_batch(dt, price, capped_lot, parking_lot_size, cycles, False)

# Plots

plt.figure(6)
plt.hist(income, edgecolor='k')
plt.ylabel('Count')
plt.xlabel('Daily Income')
plt.savefig('Figures\Income_Counts_dt_1.tiff', dpi=300)

plt.figure(7)
plt.hist(income, density=True, edgecolor='k', label='Data')
plt.plot(x_i, kde_i(x_i), 'r--')
plt.plot(x_i, p_i, 'k-')
plt.xlabel('Daily Income')
plt.ylabel('Probability Density')
plt.legend(('Data','KDE','Norm PDF'))
plt.savefig('Figures\Income_PDF_dt_1.tiff', dpi=300)

plt.figure(8)
plt.hist(max_cars, edgecolor='k')
plt.xlabel('Peak Car Total')
plt.ylabel('Count')
plt.savefig('Figures\Max_Cars_Counts_dt_1.tiff', dpi=300)

plt.figure(9)
plt.hist(max_cars, density=True, edgecolor='k', label='Data')
plt.plot(x_c,kde_c(x_c),'r--')
plt.plot(x_c, p_c, 'k', linewidth=2)
plt.xlabel('Peak Car Total')
plt.ylabel('Probability Density')
plt.legend(('Data','KDE','Norm PDF'))
plt.savefig('Figures\Max_Cars_PDF_dt_1.tiff', dpi=300)

"""

   Test 3: Run a batch with an even smaller time interval

"""


# Hyper parameters

# Time differential in hours
dt = 1/2

# Income per car
price = 10

# Is the size of the parking lot capped?
capped_lot = False
# If capped, what is the parking lot size?
parking_lot_size = 8

# Number of days in cycle
cycles = 1000

# Run batch
income, max_cars, mu_i, std_i, x_i, p_i, kde_i, mu_c, std_c, x_c, p_c, kde_c = run_batch(dt, price, capped_lot, parking_lot_size, cycles, False)

# Plots

plt.figure(10)
plt.hist(income, edgecolor='k')
plt.xlabel('Daily Income')
plt.ylabel('Count')
plt.savefig('Figures\Income_Counts_dt_0p5.tiff', dpi=300)

plt.figure(11)
plt.hist(income, density=True, edgecolor='k', label='Data')
plt.plot(x_i, kde_i(x_i), 'r--')
plt.plot(x_i, p_i, 'k-')
plt.xlabel('Daily Income')
plt.ylabel('Probability Density')
plt.legend(('Data','KDE','Norm PDF'))
plt.savefig('Figures\Income_PDF_dt_0p5.tiff', dpi=300)

plt.figure(12)
plt.hist(max_cars, edgecolor='k')
plt.xlabel('Peak Car Total')
plt.ylabel('Count')
plt.savefig('Figures\Max_Cars_Counts_dt_0p5.tiff', dpi=300)

plt.figure(13)
plt.hist(max_cars, density=True, edgecolor='k', label='Data')
plt.plot(x_c,kde_c(x_c),'r--')
plt.plot(x_c, p_c, 'k', linewidth=2)
plt.xlabel('Peak Car Total')
plt.ylabel('Probability Density')
plt.legend(('Data','KDE','Norm PDF'))
plt.savefig('Figures\Max_Cars_PDF_dt_0p5.tiff', dpi=300)

# For this case, we also want to calculate the 90% probability.
x_c_0p5 = np.linspace(0,200,10001)
kde_c_0p5 = kde_c

# Calculate the cumulative integration
F_c_0p5 = cumtrapz(kde_c_0p5(x_c_0p5), x_c_0p5, initial=0)

# Calculate the number of spots needed for 90%.
i_90pcent = sum(F_c_0p5<0.9) - 1
x_c_0p5_90pcent = x_c_0p5[i_90pcent]
print(x_c_0p5_90pcent)
print(max(x_c_0p5))

# Plot
plt.figure(9999)
plt.plot(x_c_0p5, F_c_0p5, 'b-')
plt.plot(x_c_0p5, kde_c_0p5(x_c_0p5), 'r--')
plt.plot(x_c_0p5[i_90pcent], F_c_0p5[i_90pcent], 'bo')
plt.xlabel('Peak Car Total')
plt.ylabel('Probability')
plt.legend(('Cumulative Density', 'Probability Density Function'))
plt.grid()
plt.savefig('Figures\Probability_for_dt_0p5.tiff', dpi=300)



"""

   Test 4: Run the model with several different times

"""

# Hyper parameters

# Time differential in hours
# Since we're stuck with integer intervals, the dt's must be divisible into 12
dt = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]

# Income per car
price = 10

# Is the size of the parking lot capped?
capped_lot = False
# If capped, what is the parking lot size?
parking_lot_size = 8

# Number of days in cycle
cycles = 1000

# Results
income_average = np.zeros_like(dt)
max_cars_average = np.zeros_like(dt)
max_cars_max = np.zeros_like(dt)
max_cars_min = np.zeros_like(dt)

for i in range(0,len(dt)):
    
    # Run batch
    income, max_cars, mu_i, std_i, x_i, p_i, kde_i, mu_c, std_c, x_c, p_c, kde_c = run_batch(dt[i], price, capped_lot, parking_lot_size, cycles, False)
    
    income_average[i] = np.mean(income)
    max_cars_average[i] = np.mean(max_cars)
    max_cars_max[i] = np.max(max_cars)
    max_cars_min[i] = np.min(max_cars)

plt.figure(14)
plt.plot(dt, income_average)
plt.xlabel('Time Step Size')
plt.ylabel('Average Income')
plt.ylim([0, 1.1*np.max(income_average)])
plt.grid()
plt.savefig('Figures\Income_vs_dt.tiff', dpi=300)

plt.figure(15)
plt.plot(dt, max_cars_average,'-')
plt.plot(dt, max_cars_max,'-')
plt.plot(dt, max_cars_min,'-')
plt.xlabel('Time Step Size')
plt.ylabel('Cars')
plt.legend(('Average Max Cars', 'Max Max Cars', 'Min Max Cars'))
plt.ylim([0, 1.1*np.max(max_cars_max)])
plt.grid()
plt.savefig('Figures\Max_Cars_vs_dt.tiff', dpi=300)


"""

   Test 5: Run the model with capped lot and different lot sizes

"""

# Hyper parameters

# Time differential in hours
# Since we're stuck with integer intervals, the dt's must be divisible into 12
dt = 0.5

# Income per car
price = 10

# Is the size of the parking lot capped?
capped_lot = True
# If capped, what is the parking lot size?
parking_lot_size = np.linspace(5,160,156)

# Number of days in cycle
cycles = 1000

# Results
income_average = np.zeros_like(parking_lot_size)
max_cars_average = np.zeros_like(parking_lot_size)

for i in range(0,len(parking_lot_size)):
    
    # Run batch
    income, max_cars, mu_i, std_i, x_i, p_i, kde_i, mu_c, std_c, x_c, p_c, kde_c = run_batch(dt, price, capped_lot, parking_lot_size[i], cycles, False)
    
    income_average[i] = np.mean(income)
    max_cars_average[i] = np.mean(max_cars)

plt.figure(16)
plt.plot(parking_lot_size, income_average)
plt.xlabel('Parking Lot Size')
plt.ylabel('Average Income')
plt.ylim([0, 1.1*np.max(income_average)])
plt.grid()
plt.savefig('Figures\Income_vs_parking_lot_size.tiff', dpi=300)

plt.figure(17)
plt.plot(parking_lot_size, max_cars_average,'-')
plt.xlabel('Parking Lot Size')
plt.ylabel('Average Max Cars')
plt.legend(('Average Max Cars', 'Max Max Cars', 'Min Max Cars'))
plt.ylim([0, 1.1*np.max(max_cars_average)])
plt.grid()
plt.savefig('Figures\Max_Cars_vs_parking_lot_size.tiff', dpi=300)

"""

Addendum: spots required for 100% servicing

"""

Worst_Case = np.array([0,0,0,0,0])
Worst_Case[1] = 150-5
Worst_Case[2] = Worst_Case[1] + 75-10
Worst_Case[3] = Worst_Case[2] + 60-40
Worst_Case[4] = Worst_Case[3] + 30-50

print(max(Worst_Case))

Time = [0,3,6,9,12]

plt.figure(18)
plt.plot(Time, Worst_Case)
plt.xlabel('Time (hours)')
plt.ylabel('Maximum Parked Cars: Worst Case')
plt.grid()
plt.savefig('Figures\Max_Cars_100pcent.tiff', dpi=300)

plt.show()
