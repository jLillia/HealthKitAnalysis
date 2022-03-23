#!/usr/bin/env python
# coding: utf-8

# ## Importing HealthKit Data
# 

# In[1]:


#Reading the file and converting it as a dict. 
import pandas as pd
import xmltodict

input_path = 'export.xml'

with open(input_path, 'r') as xml_file:
    input_data = xmltodict.parse(xml_file.read())


# The structure of the exported XML file is extensive, and the only dictionairies needed for this analysis are the records - a complete list of all recorded health data e.g. body mass, dietary metrics etc, the workouts - the full list of workouts recorded on an Apple Watch and the type of activity, and lastly, the Activity log, which records the three ring metrics on an Apple Watch (Activity, Exercise and Stand).`

# In[2]:


#Records list for general health data & imported as Pandas Data Frame
records_list = input_data['HealthData']['Record']
df_records = pd.DataFrame(records_list)
stepper = df_records


# In[3]:


#Workout list for workout data
workouts_list = input_data['HealthData']['Workout']
df_workouts = pd.DataFrame(workouts_list)


# In[4]:


# Activity summary
activity_summary = input_data['HealthData']['ActivitySummary']
df_activity_summary = pd.DataFrame(activity_summary)


# ## EDA and Cleaning
# 
# We'll start with the dataframe we created for the records list.

# In[5]:


df_records.head()


# The first five rows show us that there are nulls, awkard header row names, and an appended string 'HKQuantityTypeIdentifier' to the '@type' column. These will all need to be cleaned. The info() command shows us that all fields are treated as objects, so date columns will need to be recasted to datetime64 datatypes, and value to numeric datatypes.

# In[6]:


df_records.info()


# In[7]:


df_records.isna().sum()


# In[8]:


## RECORDS CLEAN

# Clean heading names by removing the @ that begins each header
df_records.columns = df_records.columns.str.replace('^@','', regex=True)

# Specify columns that we want to keep
column_names = ['type', 'sourceName','value','unit','startDate','endDate']
df_records = df_records.reindex(columns=column_names)

# Change to actual date time 
ymdformat = '%Y-%m-%d %H:%M:%S'
for col in ['startDate', 'endDate']:
    df_records[col] = pd.to_datetime(df_records[col],format = ymdformat)

# Value should be numerical, change to NA if the classifier fails
df_records['value'] = pd.to_numeric(df_records['value'], errors='coerce')

df_records.info()


# In[9]:


# Filter records inside of Game Changers date
df_records = df_records[(df_records.startDate>="2020-10-12")]
df_records = df_records[(df_records.startDate<="2020-12-06")]

for col in ['startDate', 'endDate']:
    df_records[col] = pd.to_datetime(df_records[col]).dt.date

# Clean up and reduce type names
df_records['type'] = df_records['type'].str.replace('HKQuantityTypeIdentifier', '')
df_records['type'] = df_records['type'].str.replace('HKCategoryTypeIdentifier', '')

# some records do not measure anything, just count occurences
# filling with 1.0 (= one time) makes it easier to aggregate
df_records['value'] = df_records['value'].fillna(1.0)

# We need a column to calculate duration of events for Apple Watch exercises
df_records['duration'] = df_records['endDate']-df_records['startDate']


# In[10]:


df_records.info()


# We then need to explore the types of values that are recorded, and then remove any values that won't contribute to this data analysis.

# In[11]:


df_records.groupby('type').size()


# In[12]:


# The dietary values are mostly generated for MyFitnessPal, a meal-logging tracker. I wouldn't say that the 
# data generated is accurate for all entries so the only interesting value is DietaryEnergyConsumed. The rest
# can be removed.

# Create a list of Dietary values to remove
dietaryValues = ['DietaryFatPolyunsaturated', 'DietaryFatMonounsaturated', 'DietaryFatSaturated', 
          'DietaryCholesterol', 'DietarySodium','DietaryCarbohydrates','DietaryFiber',
         'DietarySugar','DietaryProtein','DietaryVitaminC',
         'DietaryIron','DietaryPotassium', 'DietaryCalcium','DietaryFatTotal']

# Drop rows that contain any value in the list
df_records = df_records[df_records.type.isin(dietaryValues) == False]

# Define any extra values that won't be useful for analysis, these values could be useful in different analyses
# but this analysis is driven by exercise and energy consumption/expenditure. 
extraValues = ['Height','BodyFatPercentage','LeanBodyMass','NumberOfTimesFallen','SwimmingStrokeCount',
              'EnvironmentalAudioExposure','WalkingDoubleSupportPercentage','SixMinuteWalkTestDistance',
              'StairAscentSpeed','StairDescentSpeed','AppleStandHour','MindfulSession','AudioExposureEvent',
              'HandwashingEvent','HeartRateVariabilitySDNN', 'DistanceCycling','DistanceSwimming',
              'HeadphoneAudioExposure','AppleStandTime','SleepAnalysis','HighHeartRateEvent','FlightsClimbed']

# Drop rows that contain any value in the list

df_records = df_records[df_records.type.isin(extraValues) == False]


# In[13]:


df_records.groupby('type').size()


# Similar cleaning and analysis is then needed for the workouts and activity lists. 

# In[14]:


df_workouts.head()


# In[15]:


df_workouts.info()


# In[16]:


df_workouts.isna().sum()


# In[17]:


# Clean heading names by removing the @ that begins each header
df_workouts.columns = df_workouts.columns.str.replace('^@','',regex=True)

# Clean up and reduce type names
df_workouts['workoutActivityType'] = df_workouts['workoutActivityType'].str.replace('HKWorkoutActivityType', '')

# Change to actual date time format
ymdformat = '%Y-%m-%d %H:%M:%S %z'
for col in ['startDate', 'endDate']:
    df_workouts[col] = pd.to_datetime(df_workouts[col],format = ymdformat)

# Filter records inside of Game Changers date  
df_workouts = df_workouts[(df_workouts.startDate>="2020-10-12")]
df_workouts = df_workouts[(df_workouts.startDate<="2020-12-06")]

# Duration, total energy burn and total distance should be numerical, change to NA if the classifier fails
for col in ['duration','totalDistance','totalEnergyBurned']:
    df_workouts[col] = pd.to_numeric(df_workouts[col], errors='coerce')
    # some records do not measure anything, just count occurences
    # filling with 1.0 (= one time) makes it easier to aggregate
    df_workouts[col] = df_workouts[col].fillna(0.0)
 
# All duration is in minutes. Can be removed
# All distance is in kms. Can be removed
# Total energy burned is kcals. Can be removed
# Other columns in the below list are unnecessary for this analysis
column_names = ['creationDate','durationUnit', 'totalDistanceUnit','totalEnergyBurnedUnit',
               'sourceVersion','MetadataEntry','WorkoutRoute','WorkoutEvent','device']

df_workouts = df_workouts.drop(columns = column_names)

# Need to explore types of workouts, and change the values if they're incorrect
df_workouts['workoutActivityType'].unique()


# In[18]:


df_workouts['workoutActivityType'].replace("CrossTraining", "Cross Training",inplace=True)
df_workouts['workoutActivityType'].replace("TraditionalStrengthTraining", "Traditional Strength Training",inplace=True)
df_workouts['workoutActivityType'].replace("Other", "Personal Training Program",inplace=True)
# 'Other' workouts were logged by my personal trainer's app, and need to be changed.


# In[19]:


df_workouts.head()


# In[20]:


df_activity_summary.head()


# In[21]:


df_activity_summary.info()


# In[22]:


df_activity_summary.isna().sum()


# In[23]:


## ACTIVITY CLEAN
df_activity_summary.columns = df_activity_summary.columns.str.replace('^@','',regex=True)

# Change to actual date time format
df_activity_summary['dateComponents'] = pd.to_datetime(df_activity_summary['dateComponents'])

# Remove unneccessary column names
column_names = ['activeEnergyBurnedUnit','appleMoveTime','appleMoveTimeGoal']
df_activity_summary = df_activity_summary.drop(columns = column_names)

# Duration, total energy burn and total distance should be numerical, change to NA if the classifier fails
for col in ['activeEnergyBurned','activeEnergyBurnedGoal','appleExerciseTime','appleExerciseTimeGoal','appleStandHours','appleStandHoursGoal']:
    df_activity_summary[col] = pd.to_numeric(df_activity_summary[col], errors='coerce')
    # some records do not measure anything, just count occurences
    # filling with 1.0 (= one time) makes it easier to aggregate
    df_activity_summary[col] = df_activity_summary[col].fillna(0.0)

# Filter to dates inside of Game Changers
df_activity_summary = df_activity_summary[(df_activity_summary.dateComponents>="2020-10-12")]
df_activity_summary = df_activity_summary[(df_activity_summary.dateComponents<="2020-12-06")]


# ## Analysis
# 
# The first element that I was interested in looking at was Active Energy Burned. I expected that it would start low and then grow steadily across the 8 weeks.

# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate rolling means to idenitify trends
rolling_mean = df_activity_summary['activeEnergyBurned'].rolling(window=3).mean()
rolling_mean_2 = df_activity_summary['activeEnergyBurned'].rolling(window=7).mean()

fig, ax = plt.subplots(figsize=(20,6))
chart = plt.bar(df_activity_summary.dateComponents, df_activity_summary.activeEnergyBurned,color='silver')
chart2 = sns.lineplot(x=df_activity_summary.dateComponents, y =rolling_mean,color='orange', lw=3, marker="o", mec="orange",label='3-Day Rolling Average')
chart3= sns.lineplot(x=df_activity_summary.dateComponents, y =rolling_mean_2,color='darkorchid',lw=3, marker="o", mec="darkorchid",label='7-Day Rolling Average')

ax.set_title("Active Energy Burned during Fitness Playground: Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Active Energy Burned (kCal)")

ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602]) #Each Monday

plt.ylim(0,3000)


# This graph shows an increase in activity as the dates increase. Sundays were often the lowest day of activity of the week as I was either taking those as rest days, or only completing low intensity cardio to get in an extra counted session.
# 
# The rolling 3-day and 7-day means grow slowly...
# 
# 

# The next trend I'm interested in exploring is changes in heart rat eacorss the challenge's date range. I expect that my basal heart rate may be lower than it was at the start of the challenge, and that max heart rate would consistenly be high.

# In[25]:


heart_rate = df_records[(df_records.type=='HeartRate')]
resting_rate = df_records[(df_records.type=='RestingHeartRate')]

resting_rate = resting_rate.sort_values(by=['startDate'])

aggregations = {
    'value':'mean'
}

avg_hr = heart_rate.groupby('startDate').agg(aggregations)


fig, ax = plt.subplots(figsize=(20,6))
chart = plt.scatter(heart_rate['startDate'], heart_rate['value'],alpha=0.1,color='silver')
chart2 = plt.plot(resting_rate['startDate'],resting_rate['value'],color='darkorchid',marker='.',mec="darkorchid",label='Resting Heart Rate')
chart3 = plt.plot(avg_hr.index, avg_hr,color='orange',lw=3,marker="o",mec="orange",label='Average Heart Rate')

ax.set_title("Heart Rate(source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Heart Rate (BPM)")
plt.legend(loc='upper right')
plt.ylim(0,max(heart_rate['value']))

ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])

fig, ax = plt.subplots(figsize=(20,6))
chart = plt.scatter(heart_rate['startDate'], heart_rate['value'],alpha=0.1,color='gray')
chart2 = plt.plot(resting_rate['startDate'],resting_rate['value'],color='darkorchid',marker=".",mec="darkorchid")


ax.set_title("Heart Rate(source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Active Energy Burned (kCal)")


ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])

plt.ylim(45,75)


# Next recorded data type I figure could be interesting is the step count I was taking per day. This data will need to be compared to all step count data collected by my Apple Watch to gain context for analysing the difference in step counts.

# In[26]:


step_count = df_records[(df_records.type=='StepCount')]

# Aggregrate the count by start date
aggregations = {'value':'sum'}
step_count = step_count.groupby('startDate').agg(aggregations)

rolling_mean = step_count.rolling(window=3).mean()
rolling_mean_2 =  step_count.rolling(window=7).mean()

fig, ax = plt.subplots(figsize=(20,6))
chart = plt.plot(step_count.index, step_count,color='silver') # ATTEMPT TO MAKE THIS BAR GRAPH
chart2 = plt.plot(step_count.index,rolling_mean,color='orange',lw=2, label='3-Day Rolling Average')
chart3= plt.plot(step_count.index, rolling_mean_2,color='red',lw=3, marker = 'o',label='7-Day Rolling Average')

ax.set_title("Daily Number of Steps Taken during Fitness Playground: Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Active Energy Burned (kCal)")
plt.ylim(0,40000)
ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])
plt.legend(loc='upper right')


# In[27]:


# As the dates have already been filtered out during cleaning, we need to pull the data and re-clean it

# Then we only need step count data collected by my Apple Watch, so we'll filter out unnecessary columns,
# clean up the Type column again, then filter out data collected my Milhouse (my iPhone) and AllTrails,
# and recast date and value datatypes
column_names = ['type', 'sourceName','value','startDate']
stepper = stepper.reindex(columns=column_names)

stepper.columns = stepper.columns.str.replace('^@','', regex=True)
stepper['type'] = stepper['type'].str.replace('HKQuantityTypeIdentifier', '')
stepper['type'] = stepper['type'].str.replace('HKCategoryTypeIdentifier', '')
stepper = stepper[(stepper.type=='StepCount')]

extraValues = ['Milhouse','AllTrails']
stepper = stepper[stepper.sourceName.isin(extraValues) == False]

stepper['startDate'] = pd.to_datetime(stepper['startDate'])
stepper['value'] = pd.to_numeric(stepper['value'])
stepper.groupby('startDate').sum()


# My Apple Watch was purchased in September 2019, so we'll count data from this day onwards
stepper = stepper[(stepper.startDate>="2019-09-01")]
stepper['startDate'] = pd.to_datetime(stepper['startDate']).dt.date

aggregations = {'value':'sum'}
full_step_count = stepper.groupby('startDate').agg(aggregations)

rolling_mean_3 = full_step_count['value'].rolling(window=7).mean()
rolling_mean_4 =  full_step_count.rolling(window=100).mean() # MAYBE 30 DAYS


# In[28]:


fig, ax = plt.subplots(figsize=(20,6))
chart = plt.plot(full_step_count.index, full_step_count,color='silver') # ATTEMPT BAR
chart2 = plt.plot(full_step_count.index,rolling_mean_3,color='red', label='7-Day Rolling Average')
chart3= plt.plot(full_step_count.index, rolling_mean_4,color='orange',label='100-Day Rolling Average')

ax.set_title("Daily Number of Steps Taken during Fitness Playground: Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Active Energy Burned (kCal)")
plt.ylim(0,40000)
ax.xaxis.grid()
plt.legend(loc='upper right')


# In[29]:


fig, ax = plt.subplots(figsize=(20,6))
chart = plt.plot(full_step_count.index,rolling_mean_3,label='Full step count data',color='red')

chart3= plt.plot(step_count.index, rolling_mean_2,color='orange',label='Game Changers')
plt.xlim(18547.,18602.)
ax.xaxis.grid()
plt.ylim([0,30000])
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])
plt.legend(loc='upper right')
ax.set_title("Rolling 7-day mean comparisons (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Total Step Count")


# We can see here that comparing rolling averages collected from data inside the Game Changers challenge vs total data collect by my Apple watch follow similar trends, however most Game Changers dates are 10k steps about the average.

# Next I want to look at Body Mass, as this was what I noticed changed most in my body during the challenge. It moved a lot through out the challenge, and I'm interested in exploring if activity levels could be associated with the change, so I'll plot both on the same axis to explore this.

# In[30]:


bodyMass = df_records[(df_records.type=="BodyMass")]
bodyMass['startDate'] = pd.to_datetime(bodyMass['startDate']).dt.date
                                     
aggregations = {'value':'min'}
bodyMass = bodyMass.groupby('startDate').agg(aggregations)

fig, ax = plt.subplots(figsize = (20,6))

chart = plt.bar(df_activity_summary.dateComponents, df_activity_summary.activeEnergyBurned,color='silver')
ax2 = ax.twinx()
plt.scatter(bodyMass.index,bodyMass,color='red')
plt.plot(bodyMass.index,bodyMass,color='red')
ax.set_title("Body Mass changes during Fitness Playground: Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Body Mass (kg)")
plt.ylim(85,96)
ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])
plt.legend(loc='upper right')


# I'm struggling to identify a pattern here. Loss of body mass is often noted as a negative total daily energy expenditure (TDEE), i.e you have burned more calories than you have consumed. There are several calculators availble throughout the internet to roughly estimate your body's basal metabolism as a baseline. They use height, weight, gender and perceived activity level to perform these calculations. My basal rate at the time was calculated as 1,939 calories, but the level of exercise I was completing per day meant that my TDEE was approximately 3000 calories per day. So let's see if we can look into this a little more and see I was under my TDEE each day. 

# In[31]:


calories_consumed = df_records[(df_records.type=="DietaryEnergyConsumed")]

total_calories_consumed = calories_consumed.groupby('startDate').sum()

calorie_dates = total_calories_consumed.index

fig, ax = plt.subplots(figsize = (20,6))

chart=plt.bar(calorie_dates,total_calories_consumed['value']/4.184,label='Energy Consumed',color='mediumturquoise')
chart2=plt.bar(calorie_dates,df_activity_summary.activeEnergyBurned[0:55],bottom=total_calories_consumed['value']/4.184,label='Active Energy',color='orange')
plt.legend(loc='upper right')
ax.set_title("Active Energy Consumed/Burned during Fitness Playground: Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Energy Burned (kCal))")
ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])


# In[54]:


ate = total_calories_consumed['value']/4.184 # Converting for kilojoules to calories
worked = df_activity_summary.activeEnergyBurned[0:55]

basal_rate = 1939
worked.index = ate.index
total = ate-worked+basal_rate

fig, ax = plt.subplots(figsize=(20,6))
                       
plt.axhline(y=1939,color='k',linestyle='-.')       
plt.axhline(y=3000,color='k',linestyle='-.')
plt.bar(total.index,total,color='silver')
plt.ylim(0,3500)
ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])
ax.set(xlabel="Date",ylabel="Energy Burned (kCal))")
ax.set_title("Net calories consumed during Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})

ax2 = ax.twinx()
plt.scatter(bodyMass.index,bodyMass,color='red')
plt.plot(bodyMass.index,bodyMass,color='red')
ax.set_title("Body Mass changes during Fitness Playground: Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':20})
ax.set(xlabel="Date",ylabel="Body Mass (kg)")
plt.ylim(85,96)
ax.xaxis.grid()
ax.set_xticks([18547.,18554.,18561.,18568.,18575.,18582.,18589.,18596.,18602])
#plt.legend(loc='upper right')


# We can see here that days where my TDEE was greater than 3000, it trends such that I am increasing in weight, whereas days where my TDEE was below basal, there's a noticable trend of loss in body mass on the next few days. 

# Next, let's have a look a look at the type of workouts that I was completing during the challenge. 

# In[33]:


df_workouts.head()


# In[34]:


# Aggregate the workout data to explore differences
aggregations = {
    'workoutActivityType':'count',
    'duration':[min,max,'mean',sum],
    'totalEnergyBurned':[min,max,'mean',sum]
}

workouts_agg = df_workouts.groupby('workoutActivityType').agg(aggregations)

workouts_agg.columns = ["_".join(x) for x in workouts_agg.columns.ravel()] 
#Without ravel, we have headers that are identical, making it impossible to wrangle. It's an old command, but
# necessary for this type of operation. There's definitely a chance it won't run on future versions, so be wary

workouts_agg


# In[55]:


fig, ax = plt.subplots(figsize=(15,6))
chart = plt.bar(workouts_agg.index,workouts_agg['duration_mean'])
plt.xlabel('\nActivity Type')
plt.ylabel('Duration (min)')
ax.set_title("Average time of Workout Activity Type during Game Changers (source: Apple Health Data)", loc='left', fontdict={'fontsize':16})

fig, ax = plt.subplots(figsize=(15,6))
chart = plt.bar(workouts_agg.index,workouts_agg['totalEnergyBurned_mean'])
plt.xlabel('\nActivity Type')
plt.ylabel('Active Energy Burn (kCal)')
ax.set_title("Average calories burned by Workout Activity Type during Game Changers(source: Apple Health Data)", loc='left', fontdict={'fontsize':16})


# In[36]:


import numpy as np

pied = df_workouts.groupby('workoutActivityType').size()
blanks = ['']*len(pied.index)

fig, ax = plt.subplots(figsize = (10,10))
plt.axis('off')
with plt.style.context({"axes.prop_cycle" : plt.cycler("color", plt.cm.tab20c.colors)}):
    ax = fig.add_subplot(111)
    wedges, texts,autotexts = ax.pie(pied,labels = blanks,autopct='%1.1f%%',startangle=-45)
    plt.axis('off')

    
bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.82)
kw = dict(arrowprops=dict(arrowstyle="->"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate((pied.index[i]), xy=(x, y), xytext=(1.15*np.sign(x), 1.3*y),
                horizontalalignment=horizontalalignment, **kw)


ax.set_title("Distribution of exercises during Game Changers", fontdict={'fontsize':16})

plt.show()


# In[37]:


workouts_agg


# In[38]:


print("Fitness Playground: Game Changers Workout Statistics")
print(f"{workouts_agg.workoutActivityType_count.sum()} workouts")

print(f"\nTotal exercise time {workouts_agg['duration_sum'].sum():.2f} minutes ({workouts_agg['duration_sum'].sum()/60:.2f} hours) ")
print(f"Calories burned : {workouts_agg['totalEnergyBurned_sum'].sum():.2f} kcal")

print("\nAverage per workout")
print(f"Average exercise time: {workouts_agg.duration_mean.mean():.2f} minutes")
print(f"Average calories burned: {workouts_agg.totalEnergyBurned_mean.mean():.2f} kcal")


# I was expecting that my rock climbing exercise time and activity would be the greatest, it's the only social exercise I do, and we go until our hands can't hold on anymore. 
# 
# Overall, this analysis was really interesting and very fun for me to complete. I've now got a better understanding of what my body was doing, what it was capable of, while also improving my own python and analytics skills. I'm excited to do more of this, and will start logging my data again dilligently so that I can expand on this analysis for further analysis!

# In[ ]:




