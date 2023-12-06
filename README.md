# IRWA-2023
Members: u242781-u192945-u190355

# IRWA Project 2023 - Part 1
In the project's first part, we focused on understanding and processing the Russia-Ukraine war tweets dataset we created from the JSON file. This involved pre-processing the data, including extracting important details like the tweets' content, date, hashtags, likes, retweets, and URLs. We also applied text cleaning techniques to the tweets' text content, such as removing URLs, converting text to lowercase, eliminating punctuation, and handling special characters. Username and words starting with '#' were preserved for future analysis. Additionally, we changed the date original structure to a pandas datetime object to improve time-based analysis and we also created a mapping between tweet and document IDs for future evaluation.

To execute and run the notebook with Google Colab: 
- Place the Python file "IRWA_2023_part_1.ipynb," the JSON file "Rus_Ukr_war_data.json," and the "Rus_Ukr_war_data_ids.csv" file in the same directory.
- After completing this setup, open the Python notebook.
- Change the path to match the path of the files mentioned before in your directory ("Rus_Ukr_war_data.json" and "Rus_Ukr_war_data_ids.csv").
- Execute each step.
Additionally, we've included a step for installing the "demoji" package, which will be used later in the code to remove emojis from the text.

# IRWA Project 2023 - Part 2
In the project's second part, our goal was to build an inverted index to be able to find relevant information within the collection of tweets and also, to evaluate them through queries and rankings. 

To execute and run the notebook with Google Colab:
- Place the Python file "IRWA_2023_part_2.ipynb," the JSON file "Rus_Ukr_war_data.json", the "Rus_Ukr_war_data_ids.csv" file and the “Evaluation_gt.csv” file in the same directory.
- After completing this setup, open the Python notebook.
- Change the path to match the path of the files mentioned before in your directory ("Rus_Ukr_war_data.json", "Rus_Ukr_war_data_ids.csv" and “Evaluation_gt.csv”).
- Execute each step.
  Additionally, we've included a step for installing the "demoji" package, which will be used later in the code to remove emojis from the text and might not be installed.

# IRWA Project 2023 - Part 3
In the third part of the project, the goal is to rank the tweets based on relevance using the BM25 ranking algorithm, considering the tweets' popularity (likes and retweets), and the Word2Vector model. The comparison of these rankings aims to determine which algorithm performs better.

To execute and run the notebook with Google Colab:
- Place the Python file "IRWA_2023_part_3.ipynb," the JSON file "Rus_Ukr_war_data.json", the "Rus_Ukr_war_data_ids.csv" file and the “Evaluation_gt.csv” file in the same directory.
- After completing this setup, open the Python notebook.
- Change the path to match the path of the files mentioned before in your directory ("Rus_Ukr_war_data.json", "Rus_Ukr_war_data_ids.csv" and “Evaluation_gt.csv”).
- Execute each step.

# IRWA Project 2023 - Part 4
In the fourth and last part of the project, the primary goal of is to establish the user interface (UI) for our Search Engine. This comprehensive web application integrates all previous components, enabling users to search queries using a specified algorithm and view related tweets with detailed information. Additionally, the project introduces Statistics, Dashboard, and Sentiment tabs to analyze and track user interactions

To execute and run our IU, follow these steps:
- Place the project folder "search-engine-web-app-main" in your computer.
- Then, open the project folder in Visual Studio Code or a similar platform of your choice.
- Open the terminal in your chosen development environment.
- Execute the following command to initiate the web application: python .\web_app.py

Our application offers several features to explore:
- Search Engine: Utilize different algorithms to query information about the Russian-Ukranian War.
- Tweet Details: Access more detailed information about the search results.
- Statistics Tab: Analyze statistical data for insights into the users interactions.
- Dashboard: Track real-time data and visualize the search engine's performance.
- Sentiment Analysis: Explore this tab to understand the emotional tone behind sentences, words and tweets.
