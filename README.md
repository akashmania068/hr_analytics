# **_HR_Analytics_**

Predicting that the selected candidates will join the company or not.

# Client Background :-

Client is a Staffing recruitment company which recruits IT Enabled Skills employees for IT MNC Companies
for Contract basis work. Some time contract position will be for min of 6 months/12 months and based on
their performance employees will be converted in to their regular pay toll of the staffing company and the
recruited employees has to work in the client location and the staffing company will charge a fixed amount
to the client regardin the employee resource. The main focus of the staffing company is to recruit more
employees based on the client skill requirement and get more commission from the client.

# Problem Statement :-

1. When 100 employees receive the offer letter through the Staffing company only 67 employees (67%) joining
the client location for work. 33% of employees are accepting the offer but not turning for the job.
Staffing company lose their commission and the good will from their client because of this 33% employees
who accepts the job and not turning for the work. Staffing company wants to increase the
joining ratio (ratio between number of employee joined the work and the number of employee who got the offer)
which will increase their revenue and the respect which will help them to charge more for their services.
The staffing company MD expects the business insights from the model so that he can pass it on to their hiring
staffs so that they can give offer only to the candidates who will most likely to join the job.

2. Given the information in the dataset, can you try to find those parameters which is impacting the amount
of hike or total salary those employees are getting?

# Deliverable :-

1. Important factors which leads for Joining(Active) & Leaving (Drop) the offer

2. Business Insights to their Hiring Staff whom they have to give the offer and to whom they should not give the offer.

3. Confussion matrix for the model validation & Accuracy

4. Consider only Joined Status = Active & Drop records for model. Joined Status = Pipeline means the candidates are yet
   to make the decision they have the some more time to join. Do not consider Joined Status = Pipeline records for analysis.

# Data Dictionary :-

Emp Id                                                        - Masked

Emp Name                                                      - Masked

Recruiter Name                                                - Masked

Company Name                                                  - Company for which Employee was hired

Interest_Reason                                               - Reason for interest for the Job filled by the candidate

Age                                                           - Age

Gender                                                        - Gender

Married                                                       - Married

Married_With_Children                                         - Married_With_Children

Sourcing Method                                               - Channel through which the candidate has applied for Job

Current Location                                              - Current Location/City

Experience Level                                              - Experience Level

Qualification                                                 - Qualification

Notice_Period_Days                                            - Notice Period in Days

Contract Duration                                             - Contract Duration in Months

Job Changes (in last 4 yrs)                                   - No of Job Changes in Last 4 Years

Employment Mode	Employment Mode (Current Job)                 - Permanent / Contract/ Unemployed/ Fresher

Offered Employment Mode	Offered Employment Mode (New Job)     - C2H (Contract to Hire ) / Contract but more than 6 months ( project will be there min 6 months contract)

Employer Category                                             - Employer Category(Current Job)

Offered Employer Category                                     - Offered Employer Category(New Job)

Size of Company                                               - Size of Company(Current Job)

Offered Size of Company                                       - Offered Size of Company (New Job)

Salary Change (CTC Amount in Rs)                              - Current Job Salary

Offered Salary Change (CTC Amount in Rs)                      - Offered Job Salary

Hike                                                          - ((Offered Job Salary - Current Job Salary ) / Current Job Salary) * 100

Career Impact                                                 - Recruiter's Feedback why candidate is interested in the job

Work Schedule Change	                                      - Current job Work Shift

Offered Work Schedule Change                                  - New Job Work Shift

Location Advantage(Proximity)                                 - Current Job Location

Offered Location Advantage(Proximity)                         - New Job Location

Skills                                                        - Job Skills

Joined Status	                                              - Active   - Accepted the Offer & Joined, 
                                                                Drop     - Accepted the Offer but Did not Joined, 
                                                                Pipeline - Accepted the Offer - Yet to make the decision (still has more days to join)
