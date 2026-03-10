As of now, we have built a pipeline that gets us from raw YOTO data into a pipeline that preprocesses it and run labram on that data (pitch data)

Now you will continue with more data to build a classifier for color detection. You will follow the way we have been working so far. (when it comes to flow of experiment)

- @data.md:1-100 @eeg_datasets_for_labram.csv:1-12 


- Get max 10 subjects's data from each of the datasets that are associated with color stimuli (2, 3, 7th ones). If there are a lot of useless data (rest data for example), try to only download the relevant data (task data)

- Build the preprocessing utilities to get the data downloaded to the right format 

- Run and validate preprocessing utilities

- Processed the datasets

- Run Labram on the processed datasets individually and gather results

- Combine datasets and run labram on all of them

- Ensure all dashboarding capabilities previously available are available for those new datasets

