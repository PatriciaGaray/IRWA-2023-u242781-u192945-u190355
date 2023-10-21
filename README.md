# IRWA-2023
Members: u242781-u192945-u190355

# IRWA Project 2023 - Part 1
In the project's first part, we focused on understanding and processing the Russia-Ukraine war tweets dataset we created from the JSON file. This involved pre-processing the data, including extracting important details like the tweets' content, date, hashtags, likes, retweets, and URLs. We also applied text cleaning techniques to the tweets' text content, such as removing URLs, converting text to lowercase, eliminating punctuation, and handling special characters. Username and words starting with '#' were preserved for future analysis. Additionally, we changed the date original structure to a pandas datetime object to improve time-based analysis and we also created a mapping between tweet and document IDs for future evaluation.

To execute and run the notebook with Google Colab: 
- Place the Python file "IRWA_2023_part_1.ipynb," the JSON file "Rus_Ukr_war_data.json," and the "Rus_Ukr_war_data_ids.csv" file in the same directory.
- After completing this setup, open the Python notebook.
- Execute each step.
Additionally, we've included a step for installing the "demoji" package, which will be used later in the code to remove emojis from the text.
