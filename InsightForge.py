#!/usr/bin/env python
# coding: utf-8

# ## Step 1. Load the data:
# 1. Loads the sales data from a CSV file
# 
# #### Data Description:
# The dataset `sales_data.csv` contains 2500 records with the following columns:
# 
# 1. `Date`: The date of the sale (format: YYYY-MM-DD).
# 2. `Product`: The name of the product sold.
# 3. `Region`: The geographical region where the sale occurred.
# 4. `Sales`: The sales amount in currency units.
# 5. `Customer_Age`: The age of the customer.
# 6. `Customer_Gender`: The gender of the customer.
# 7. `Customer_Satisfaction`: The satisfaction rating of the customer on a scale (float value).

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('sales_data.csv')


# ## Step 2: Knowledge Base Creation

# - By structuring data in a well-organized format, a knowledge base enables efficient data retrieval.
# - This is particularly important when using advanced AI techniques like Retrieval-Augmented Generation (RAG), which rely on quickly accessing relevant data to generate accurate insights on `df`.

# #### 2.1. Import necessary libraries

# In[5]:


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


# Create a `PDF Folder` which acts as a storage folder for all the reference PDFs 

# In[6]:


# list all the PDFs in the PDF Folder
os.listdir('PDF Folder')


# #### 2.2. Load PDFs from the folder using **PyPDFLoader**
# 
# This iterates over each file in the **PDF Folder** directory, checks if the file is a PDF, and uses PyPDFLoader to load the content of each PDF into the documents list.

# In[7]:


pdf_folder = 'PDF Folder'
documents = []
for file in os.listdir(pdf_folder):
    if file.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())


# #### 2.3. Split documents into chunks
# 
# **RecursiveCharacterTextSplitter** is initialized with a chunk size=1000 and overlap=200. The loaded documents are split into smaller text chunks. It also prints the total number of chunks and a sample of the first chunk.

# In[8]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)



# #### 2.4. Save processed text
# Saving the processed texts in `.pkl` format. It allows for easy reloading and use in future sessions without needing to reprocess the original documents.
# 
# `pickle` provides a reliable and efficient way to save and load complex data structures, making it a practical choice for preserving the processed text chunks in this project.
# 
# 
# 
# 
# 
# 

# In[9]:


import pickle
with open('processed_texts.pkl', 'wb') as f:
    pickle.dump(texts, f)



# ## Step 3: Creating langchain setup

# In[10]:


import numpy as np
from scipy import stats
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI


# #### 3.1. Generate Advanced Data Summary:
# 
# Create `generate_advanced_data_summary` function to perform a comprehensive analysis of the business data, including sales analysis, time-based analysis, product and regional performance, and customer insights.
# 
# **Perform the following analysis:**
# 1. **Data Preparation**:
#    Convert the 'Date' column to datetime format to enable time-based analysis.
# 
# 2. **Sales Analysis:**
#    Calculate total sales, average sale, median sale, and standard deviation of sales, providing a statistical overview of sales performance.
# 
# 3. **Time-based Analysis:**
#    Aggregates sales data by month and identifies the best and worst performing months based on sales volume.
# 
# 4. **Product Analysis:**
#    Analyze sales data by product, identifying the top-selling product by total sales value and the most frequently sold product by sales count.
# 
# 5. **Regional Analysis:**
#    Aggregates sales data by region, identifying the best and worst performing regions.
# 
# 6. **Customer Analysis:**
#     - Analyze customer satisfaction scores mean and standard deviation.
#     - Segment customers by age group and calculates average sales for each group, identifying the best-performing age group.
#     - Analyze average sales by customer gender.
#   
# 
#  Create an advanced data summary using this function. This summary serves as the input for the Retrieval-Augmented Generation (RAG) system, which combines structured data analysis with natural language generation to provide in-depth insights and recommendations.

# In[11]:


def generate_advanced_data_summary(df):
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sales Analysis
    total_sales = df['Sales'].sum()
    avg_sale = df['Sales'].mean()
    median_sale = df['Sales'].median()
    sales_std = df['Sales'].std()

    # Time-based Analysis
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month', observed=False)['Sales'].sum().sort_values(ascending=False)
    best_month = monthly_sales.index[0]
    worst_month = monthly_sales.index[-1]

    # Product Analysis
    product_sales = df.groupby('Product', observed=False)['Sales'].agg(['sum', 'count', 'mean'])
    top_product = product_sales['sum'].idxmax()
    most_sold_product = product_sales['count'].idxmax()

    # Regional Analysis
    region_sales = df.groupby('Region', observed=False)['Sales'].sum().sort_values(ascending=False)
    best_region = region_sales.index[0]
    worst_region = region_sales.index[-1]

    # Customer Analysis
    avg_satisfaction = df['Customer_Satisfaction'].mean()
    satisfaction_std = df['Customer_Satisfaction'].std()

    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, labels=age_labels, right=False)
    age_group_sales = df.groupby('Age_Group', observed=False)['Sales'].mean().sort_values(ascending=False)
    best_age_group = age_group_sales.index[0]

    # Gender Analysis
    gender_sales = df.groupby('Customer_Gender', observed=False)['Sales'].mean()

    summary = f"""
    Advanced Sales Data Summary:

    Overall Sales Metrics:
    - Total Sales: ${total_sales:,.2f}
    - Average Sale: ${avg_sale:.2f}
    - Median Sale: ${median_sale:.2f}
    - Sales Standard Deviation: ${sales_std:.2f}

    Time-based Analysis:
    - Best Performing Month: {best_month}
    - Worst Performing Month: {worst_month}

    Product Analysis:
    - Top Selling Product (by value): {top_product}
    - Most Frequently Sold Product: {most_sold_product}

    Regional Performance:
    - Best Performing Region: {best_region}
    - Worst Performing Region: {worst_region}

    Customer Insights:
    - Average Customer Satisfaction: {avg_satisfaction:.2f}/5
    - Customer Satisfaction Standard Deviation: {satisfaction_std:.2f}
    - Best Performing Age Group: {best_age_group}
    - Gender-based Average Sales: Male=${gender_sales['Male']:.2f}, Female=${gender_sales['Female']:.2f}


    Key Observations:
    1. The sales data shows significant variability with a standard deviation of ${sales_std:.2f}.
    2. The {best_age_group} age group shows the highest average sales.
    3. Regional performance varies significantly, with {best_region} outperforming {worst_region}.
    4. The most valuable product ({top_product}) differs from the most frequently sold product ({most_sold_product}), suggesting potential for targeted marketing strategies.
    """

    return summary


# In[12]:


advanced_summary = generate_advanced_data_summary(df)


# #### 3.2. Langchain Setup:
# 1. Initialize the `ChatOpenAI` model with a specific temperature setting and model name.

# In[13]:


from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")


# `scenario_template` defines the template for the prompt that will be sent to the language model. The template sets the scenario where the model acts as an "expert AI sales analyst". It includes placeholders {advanced_summary} and {question}, which will be filled with specific data summaries and questions when generating the prompt.

# In[14]:


scenario_template = """
You are an expert AI sales analyst. Use the following advanced sales data summary to provide in-depth insights and actionable recommendations. Be specific and refer to the data points provided.

{advanced_summary}

Question: {question}

Detailed Analysis and Recommendations:
"""


# 1. The `PromptTemplate` is instantiated with the `scenario_template` and the expected input variables (`advanced_summary and question`). This template will structure the input data consistently for the model.
# 
# 2. The LLMChain object called `llm_chain` is then created, linking the prompt with the chat_model. This chain is responsible for managing the flow of data from the input prompt through the model to the output response.

# In[15]:


prompt = PromptTemplate(template=scenario_template, input_variables=["advanced_summary", "question"])
llm_chain = LLMChain(prompt=prompt, llm=chat_model)


# Create `generate_insight` function that calls the run method of the `llm_chain` object, passing in the `advanced_summary` and `question` as arguments. The `llm_chain` has been previously configured with a prompt template and the GPT-3.5 Turbo model.

# In[16]:


def generate_insight(advanced_summary, question):
    return llm_chain.run(advanced_summary=advanced_summary, question=question)


# #### Test `generate_insight` function:
# - Pass a dummy question based on the dataset to test the insight generation


# ## Step 4: Prompt Chaining
# Use LangChain to create a sequential chain for analyzing sales data and generating recommendations. We'll use OpenAI's language models to process our data summary and provide insights.
# 

# In[32]:


from langchain.chains import SequentialChain


# #### 4.1. Create First chain for data analysis
# 
# 1. Data Analysis Template: Create a template to define the prompt for the language model to analyze the advanced sales data summary. This is created to provide a concise analysis of key points.
# 
# 2. Create a Data Analysis Chain: Link `data_analysis_prompt` to the chat_model through LLMChain, creating a chain that handles the data analysis task.

# In[29]:


data_analysis_template = """
Analyze the following advanced sales data summary:

{advanced_summary}

Provide a concise analysis of the key points:
"""

data_analysis_prompt = PromptTemplate(template=data_analysis_template, input_variables=["advanced_summary"])
data_analysis_chain = LLMChain(llm=chat_model, prompt=data_analysis_prompt, output_key="analysis")


# #### 4.2. Create the second chain in the sequence
# 
# 1. Create a Recommendation Template: `recommendation_template` instructs the model to generate specific recommendations based on the sales data analysis, tailored to a given question.
# 2. Create a PromptTemplate object from the LangChain library called `recommendation_prompt`. It uses the `recommendation_template` defined above. `input_variables` specifies that this prompt expects two inputs: "analysis" and "question".
# 2. Create Recommendation Chain: `recommendation_chain` should take the analysis from the previous chain and a specific question to generate recommendations

# In[30]:


# Second chain for recommendations
recommendation_template = """
Based on the following analysis of sales data:

{analysis}

Provide specific recommendations to address the question: {question}

Recommendations:
"""

recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=["analysis", "question"])
recommendation_chain = LLMChain(llm=chat_model, prompt=recommendation_prompt, output_key="recommendations")


# #### 4.3. Combine the chains
# 
# 1. Create a Sequential Chain `overall_chain`:  `SequentialChain` combines the data analysis and recommendation chains, allowing them to run in sequence. This structure takes the `advanced_summary` and `question` as inputs and returns `analysis` and `recommendations` specified in the recommendation template.
#    
# 2. Generate Chained Insight Function: Write a function `generate_chained_insight` that takes a question as input and uses the `overall_chain` to generate/return both analysis and recommendations based on the `advanced summary` and the `question`.

# In[33]:


overall_chain = SequentialChain(
    chains=[data_analysis_chain, recommendation_chain],
    input_variables=["advanced_summary", "question"],
    output_variables=["analysis", "recommendations"],
    verbose=True
)

def generate_chained_insight(question):
    try:
        result = overall_chain({"advanced_summary": advanced_summary, "question": question})
        return f"Analysis:\n{result['analysis']}\n\nRecommendations:\n{result['recommendations']}"
    except Exception as e:
        print(f"Error in generate_chained_insight: {e}")
        return f"Error: {str(e)}"


# ## Step 5: RAG System Setup:
# 
# The section integrates the `ChatOpenAI` model (GPT-3.5 Turbo) with a retriever through a `RetrievalQA` chain, enabling the system to answer questions based on the retrieved data. It also enhances responses by incorporating information from Wikipedia, facilitated by the `WikipediaAPIWrapper` and a custom search function. The script is tested by generating insights and recommendations, focusing on improving customer satisfaction trends.

# In[35]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.utilities import WikipediaAPIWrapper
from datetime import datetime


# In[36]:


# Load processed texts from pickle file
with open('processed_texts.pkl', 'rb') as f:
    texts = pickle.load(f)


# #### 5.1. Create Embeddings and Vector Store:
# 1. This step involves creating embeddings for the text documents using `OpenAIEmbeddings`.
# 2. These embeddings are stored in a FAISS vector store, which is optimized for quick similarity search.
# 3. Set up the `retriever` to retrieve the top 3 most similar documents based on similarity search.

# In[37]:


# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# #### 5.2. Create RetrievalQA chain:
# The RetrievalQA chain, `qa_chain` integrates the chat model with the retriever. It fetches relevant documents from the vector store and uses them as context for generating responses.
# Set `return_source_documents` parameter as `True` to ensure that the sources of the information are returned along with the response.

# In[38]:


# Initialize ChatOpenAI
chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# #### 5.3. Adding Wikipedia Search Functionality
# 
# 1. Define `wiki_search` function which takes a query, searches Wikipedia, and returns the content and URLs of the top results. It handles common exceptions like disambiguation errors and page errors.
# 2. Create a Wikipedia search tool, `wikipedia_tool`. It uses the `WikipediaAPIWrapper` and the `wikipedia` library to search for and retrieve content from Wikipedia. The tool is set up using the Tool class from LangChain, making it accessible for the RAG system to use Wikipedia data in enhancing responses.

# In[39]:


from langchain.tools import Tool
import wikipedia
from bs4 import BeautifulSoup

# Initialize Wikipedia API wrapper
wikipedia_wrapper = WikipediaAPIWrapper()

def wiki_search(query):
    try:
        # Use the wrapper to get content
        content = wikipedia_wrapper.run(query)

        # Use Wikipedia search to get URLs
        search_results = wikipedia.search(query, results=3)
        urls = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                # Specify the parser explicitly
                soup = BeautifulSoup(page.html(), features="lxml")
                content += f"\nTitle: {page.title}\nSummary: {soup.get_text()[:500]}...\n"
                urls.append(page.url)
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
        return {'content': content, 'urls': urls}
    except Exception as e:
        return {'content': f"An error occurred: {str(e)}", 'urls': []}

wikipedia_tool = Tool(
    name="Wikipedia Search",
    func=wiki_search,
    description="Searches Wikipedia for information"
)


# ## Step 6: Memory Integration:

# #### 6.1. Instantiate a memory:
# 
# 1. Instantiate a `ConversationBufferMemory` to store the conversation history allowing the model to reference past interactions.
# 2. Set up a `ConversationChain` with the `chat_model` and `ConversationBufferMemory()`, to enable the system to generate responses that are informed by previous context. The verbose flag ensures detailed logs of the interactions are maintained.

# In[44]:


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True
)


# #### 6.2. Generating RAG Insights with Wikipedia Integration and Conversation memory:
#  - Create a function `generate_rag_insight` that generates insights by first using the `RetrievalQA` chain to retrieve relevant documents from the vector store and use the `overall_chain` to analyze the advanced summary and the retrieved documents. The `generate_rag_insight` function updates the memory with the current context and results after generating insights.
#    
#  This function queries Wikipedia for additional content related to the question, enhancing the context with this information. Generate final insight using the combined context, and compile the sources, including both the retrieved documents and Wikipedia URLs. This integration makes for richer, more comprehensive answers by **leveraging both internal data and external knowledge from Wikipedia**.

# Functionality:
# 
# `generate_rag_insight_with_memory_seqchain` function initiates the process by using a retriever to get relevant documents based on the input question.
# It is expected to leverage a sequential chain and conversation memory to generate insights, though the rest of the function details are missing in the provided snippet.

# In[ ]:


def generate_rag_insight_with_memory_seqchain(question):
    # Use the existing retriever to get relevant documents
    retrieved_documents = retriever.get_relevant_documents(question)
    retrieved_texts = " ".join([doc.page_content for doc in retrieved_documents])

    # Use the overall_chain to analyze the advanced summary and the retrieved documents
    context = {
        "advanced_summary": advanced_summary,
        "question": question
    }
    analysis_result = overall_chain.apply([context])[0]
    analysis = analysis_result['analysis']
    recommendations = analysis_result['recommendations']

    # Get Wikipedia content and URLs related to the question
    wiki_results = wikipedia_tool.run(question)
    wiki_content = wiki_results['content']
    wiki_urls = wiki_results['urls']

    # Combine the existing context with Wikipedia results
    enhanced_context = f"{analysis}\n\nAdditional information from Wikipedia:\n{wiki_content}"
    
     # Use the enhanced context to generate the final insight
    final_result = conversation.predict(input=enhanced_context)

    insight = final_result
    sources = [doc.metadata['source'] for doc in retrieved_documents]

    # Add Wikipedia to the sources, including specific URLs
    sources.extend([f"Wikipedia: {url}" for url in wiki_urls])

    return f"Analysis:\n{analysis}\n\nRecommendations:\n{recommendations}\n\nEnhanced Insight:\n{insight}\n\nSources:\n" + "\n".join(set(sources))


# Functionality:
# 
# `generate_rag_insight` function first constructs a context from the advanced sales summary and the input question.
# It uses the `qa_chain` to process this context and retrieve relevant documents.
# Wikipedia content is fetched and combined with the initial context to enhance the overall insight.
# The `qa_chain` is then used again to generate the final insight based on this enriched context.
# The final result includes the insight and a list of sources, including Wikipedia URLs.

# In[50]:


def generate_rag_insight(question):
    # First, use the existing retriever to get relevant documents
    context = f"Advanced Sales Summary:\n{advanced_summary}\n\nQuestion: {question}"
    result = qa_chain({"query": context})

    # Get Wikipedia content and URLs related to the question
    wiki_results = wikipedia_tool.run(question)
    wiki_content = wiki_results['content']
    wiki_urls = wiki_results['urls']

    # Combine the existing context with Wikipedia results
    enhanced_context = f"{context}\n\nAdditional information from Wikipedia:\n{wiki_content}"

    # Use the enhanced context to generate the final insight
    final_result = qa_chain({"query": enhanced_context})

    insight = final_result['result']
    sources = [doc.metadata['source'] for doc in final_result['source_documents']]

    # Add Wikipedia to the sources, including specific URLs
    sources.extend([f"Wikipedia: {url}" for url in wiki_urls])

    return f"Insight: {insight}\n\nSources:\n" + "\n".join(set(sources))


# #### 6.3. Generating Insights with Conversation Memory
# 
# Define a function `generate_insight_with_memory`, that takes a question as input and uses the conversation chain to generate an insight.
# It combines the advanced sales summary and the question to provide context for the response, leveraging the memory to maintain continuity across different queries.

# In[51]:


def generate_insight_with_memory(question):
    return conversation.predict(input=f"Advanced Sales Summary:\n{advanced_summary}\n\nQuestion: {question}")


# ## Step 7: External Tool Integration and Model Evaluation:
# This involves the integration of external tools, including Wikipedia searches and data visualization, into a conversational AI model.
# The setup also includes the evaluation and monitoring of the model's performance using a combination of Retrieval-Augmented Generation (RAG) and Streamlit for a seamless user interface.

# In[54]:


import matplotlib.pyplot as plt
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.evaluation.qa import QAEvalChain


from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union


# #### 7.1. Visualizing data:
# - Define two functions to generate plots for Sales distribution by product and daily sales trend.
# - This is created as visual insights into the sales data, which can be used alongside textual analysis to give a fuller picture of business performance.
# 
# **Steps to be followed:**
# 
# *Plot for Sales distribution by product:* `plot_product_category_sales()`
# 
# 1. Use <dataset>.groupby('Product')['Sales'].sum() to calculate total sales for each product.
# 2. sort_values(ascending=False) orders the products from highest to lowest sales.
# 3. Use plt.figure(figsize=(10, 6)) to set the size of the plot.
# 4. Create the bar plot from the grouped data from **step 1.**
# 5. Set labels and rotate x-axis labels for better readability.
# 6. The plot is saved as a PNG file and the file path is returned.
# 
# *Plot for Daily Sales Trend:* `plot_sales_trend`
# 1. Use <dataset>.groupby('Date')['Sales'].sum() to calculate total sales for each date.
# 2. Use `.plot()` without specifying a kind. This defaults to a line plot, which is suitable for time series data.
# 3. Set appropriate labels and save the plot as a PNG file.

# In[55]:


def plot_product_category_sales():
    product_cat_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    product_cat_sales.plot(kind='bar')
    plt.title('Sales Distribution by Product')
    plt.xlabel('Product')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()



def plot_sales_trend():
    plt.figure(figsize=(10, 6))
    df.groupby('Date')['Sales'].sum().plot()
    plt.title('Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')


# #### 7.2. Integrating Tools with ZeroShotAgent:
# - This segment integrates various tools into a `ZeroShotAgent` setup, allowing the agent to use these tools to answer questions.
# - It sets up the agent with a prompt template and tools for generating responses, including visualizations.

# #### Create a set of tools that encapsulate the analysis functions.
# 
# **List of Tools:**
# 1. `ProductCategorySalesPlot`
# - This tool calls the `plot_product_category_sales()` function.
# This tool visualizes representation of sales across different product categories.
# 
# 
# 2. `SalesTrendPlot`
# - This tool calls the `plot_sales_trend()` function.
# - This tool visualizes how sales have trended over time.
# 
# 
# 3. `AdvancedSummary`
# - This tool returns the `advanced_summary`.
# - It provides a comprehensive textual summary of the sales data.
# - This tool provides an overview of key sales metrics and insights.
# 
# 
# 4. `RAGInsight`
# - This tool uses the `generate_rag_insight()` function.
# - It generates insights using our Retrieval-Augmented Generation (RAG) system.
# - This tool provides AI-generated insights based on both our sales data and external knowledge.
# 
# These tools provide a convenient way to access different aspects of sales analysis. They can be easily integrated into a larger system, such as a chatbot or an automated reporting tool, allowing for flexible and powerful data analysis capabilities.
# The **plotting tools** will save the plots as image files and return the file paths, while the **summary and insight tools** return text-based analysis.
# 
# **Note**: Always check the returned values to understand the output of each tool.

# In[56]:

tools = [
    Tool(
        name="ProductCategorySalesPlot",
        func=lambda x: plot_product_category_sales,
        description="Generates a plot of sales distribution by product category"
    ),
    Tool(
        name="SalesTrendPlot",
        func=lambda x: plot_sales_trend,
        description="Generates a plot of the daily sales trend"
    ),
    Tool(
        name="AdvancedSummary",
        func=lambda x: advanced_summary,
        description="Provides the advanced summary of sales data"
    ),
    Tool(
        name="RAGInsight",
        func=generate_rag_insight,
        description="Generates insights using RAG system"
    )
]


# #### 7.3. Set up a Sales Analyst Agent:

# 
# Define the prompt that will guide the agent's behavior:
# 1. Add the `prefix` that provides the agent its role and capabilities.
# 2. Add the `suffix` that provides a structure for the conversation, including placeholders for chat history, user input, and the agent's thought process.
# 3. Use `ZeroShotAgent.create_prompt()` to create a prompt that includes descriptions of all the tools created to deliver insights.

# In[57]:


prefix = """You are an AI sales analyst with access to advanced sales data and a RAG system.
Use the following tools to answer the user's questions:"""

suffix = """Begin!"

{chat_history}
Human: {input}
AI: Let's approach this step-by-step:
{agent_scratchpad}"""


prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)


# #### 7.4. Create the Agent:
# 1. Create an `LLMChain` using the previously initialized chat model and the prompt.
# 2. It is followed by creating a `ZeroShotAgent` using this chain and the tools.
# 3. Finally, create an `AgentExecutor` that can run the agent with the tools.

# In[58]:


llm_chain = LLMChain(llm=chat_model, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# #### 7.5. Run the Agent
# To use the agent, run it with a specific input:
# 
# `sample input: "Analyze our sales performance and suggest strategies for improvement. Include visualizations in your analysis."`



# ## Step 8. Model Evaluation and Monitoring:

# This section evaluates the model's performance using predefined question-answer pairs. It uses the `QAEvalChain` to compare the model's responses with the expected answers, calculating the accuracy of the model.
# 

# #### 8.1. Creating Question-Answer Pairs
# 
# <!-- 1. Create a set of question-answer pairs based on the sales data by defining a function `create_qa_pairs()`. Use them to evaluate the performance of the Agent. This process helps us understand how well the model performs on specific, factual questions about the Sales data. -->
# 
# 1. Create a set of question-answer pairs based on the sales data. Use them to evaluate the performance of the Agent. This process helps us understand how well the model performs on specific, factual questions about the Sales data.
# 
# <!-- The function contains a list of dictionaries, each containing a question and its corresponding answer. The answers are generated directly from the sales data (df). -->
# 
# **Key points:**
# 1. The questions cover different aspects of our sales data: total sales, best-performing product, and customer satisfaction.
# 2. Use f-strings to insert calculated values into the answers.

# In[60]:


qa_pairs = [
        {
            "question": "What is our total sales amount?",
            "answer": f"The total sales amount is ${df['Sales'].sum():,.2f}."
        },
        {
            "question": "Which product category has the highest sales?",
            "answer": f"The product category with the highest sales is {df.groupby('Product')['Sales'].sum().idxmax()}."
        },
        {
            "question": "What is our average customer satisfaction score?",
            "answer": f"The average customer satisfaction score is {df['Customer_Satisfaction'].mean():.2f}."
        },
        # Add more question-answer pairs as needed
    ]


# #### 8.2. Evaluate the model using QAEvalChain:
# - Define a function `evaluate_model` to evaluate the model using the Question-Answer pairs
# 
# `evaluate_model` performs the following:
# 
# 1. Creates an evaluation chain using the chat model.
# 2. Generates predictions by running each question through the agent.
# 3. Evaluates the predictions against the actual answers.
#    
# 
# **Function Breakdown:**
# 1. *Initialization:*
# Create QAEvalChain, `eval_chain` using `chat_model`. The `handle_parsing_errors=True` parameter makes it more robust to unexpected formats.
# 
# 2. *Evaluation Process:*
# `eval_chain.evaluate()` takes `qa_pairs` (ground truth), `predictions`(model outputs), `question_key`, `answer_key`, `prediction_key`.
# 
#     For each question-answer pair:
#     - It compares the predicted answer to the actual answer.
#     - It uses the language model to judge if the prediction correctly answers the question.
# 
# 3. *Output:*
# It returns a list of evaluation results, where each result includes:
# - The original question
# - The predicted answer
# - The actual answer
# - A boolean indicating if the prediction is correct

# In[61]:


def evaluate_model(qa_pairs):
    eval_chain = QAEvalChain.from_llm(chat_model, handle_parsing_errors=True)
    # Get predictions
    predictions = []
    for q in qa_pairs:
        result = agent_chain.run(input=q["question"], chat_history="", agent_scratchpad="")
        predictions.append({"question": q["question"], "prediction": result})

    # Evaluate results
    eval_results = []
    # Evaluate results
    results = eval_chain.evaluate(
        examples=qa_pairs,
        predictions=predictions,
        question_key="question",
        answer_key="answer",
        prediction_key="prediction"  # Ensure this matches the keys used in predictions
    )

    for i, result in enumerate(results):
        eval_results.append({
            "question": qa_pairs[i]["question"],
            "predicted": predictions[i]["prediction"],
            "actual": qa_pairs[i]["answer"],
            "correct": result["results"] == 'CORRECT'
        })
    return eval_results


# #### 8.3. Run evaluation

# In[62]:


eval_results = evaluate_model(qa_pairs)



# ## Evaluation in the Context of LLMOps
# 
# Evaluation is a crucial component of LLMOps. LLMOps, similar to MLOps (Machine Learning Operations), refers to the practices and technologies used to streamline and optimize the lifecycle of large language models in production environments.
# 
# ## How Evaluation Fits into LLMOps
# 
# 1. **Model Performance Monitoring**
#    - The evaluation section helps monitor the performance of the AI sales analyst agent over time.
#    - It provides quantitative metrics (like accuracy) that can be tracked across different versions or deployments of the model.
# 
# 2. **Quality Assurance**
#    - By regularly running these evaluations, we can ensure that the model maintains a certain level of quality and reliability in its responses.
#    - This is crucial for maintaining trust in the AI system, especially in a business context where decisions might be influenced by the model's outputs.
# 
# 3. **Continuous Improvement**
#    - The detailed results from our evaluation (question-by-question breakdown) can guide focused improvements to the model or its underlying knowledge base.
#    - It helps identify specific areas where the model might need fine-tuning or additional training data.
# 
# 4. **Version Control and Model Comparison**
#    - In an LLMOps pipeline, you might have multiple versions of your model. This evaluation framework allows you to compare performance across different versions.
# 
# 5. **Automated Testing**
#    - This evaluation can be integrated into an automated testing pipeline, allowing for continuous evaluation as part of the deployment process.
# 
# 6. **Domain-Specific Evaluation**
#    -The use of sales-specific questions demonstrates how LLMOps practices can be tailored to specific domains or use cases.
# 
# 7. **Feedback Loop**
#    - The results from these evaluations can feed back into the model development process, informing decisions about data collection, model architecture, or fine-tuning strategies.
# 

# ## Step 9: Model Monitoring
# 
# #### 9.1. **Create a SimpleModelMonitor: A Basic LLMOps Monitoring Tool:**
# 
# The `SimpleModelMonitor` class is a basic implementation of a monitoring system for the AI sales analyst. It's designed to track and analyze the performance of chat model over time, which is a crucial aspect of LLMOps (Large Language Model Operations).
# 

# **Key Methods:**
# 1. *Loading and Saving Logs:*
#    
#    These functions handle the persistence of log data:
#     - `load_logs()` reads existing logs from a JSON file.
#     - `save_logs()` writes the current logs back to the JSON file.
#     
# 2. *Log Interactions:*
#    
#    `log_interaction` function logs each interaction with the model, recording:
#     - The timestamp
#     - The query sent to the model
#     - The execution time of the query
# 
# 3. *Visualization:*
# 
#     `plot_execution_times` creates a plot of execution times over timestamp, providing a visual representation of the model's performance.
# 
# 4. *Performance Metrics:*
# 
#    `get_average_execution_time` calculates and returns the average execution time across all logged interactions.
# 
# #### LLMOps Relevance
# 
# 1. **Performance Monitoring:** By tracking execution times, the model's performance can be monitored over time.
# 2. **Usage Patterns:** Logging queries allows us to understand how the model is being used.
# 3. **Visualization:** The plotting function provides an easy way to visualize performance trends.
# 4. **Data Persistence:** Saving logs to a file ensures we don't lose valuable monitoring data between runs.
# 5. **Basic Analytics:** The average execution time method provides a simple performance metric.
# 
# 

# In[65]:


import json

class SimpleModelMonitor:
    def __init__(self, log_file='simple_model_monitoring.json'):
        self.log_file = log_file
        self.logs = self.load_logs()

    def load_logs(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []

    def log_interaction(self, query, execution_time):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'execution_time': execution_time
        }
        self.logs.append(log_entry)
        self.save_logs()

    def save_logs(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def plot_execution_times(self):
        timestamps = [datetime.fromisoformat(log['timestamp']) for log in self.logs]
        execution_times = [log['execution_time'] for log in self.logs]

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, execution_times, marker='o')
        plt.title('Model Execution Times')
        plt.xlabel('Timestamp')
        plt.ylabel('Execution Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('simple_execution_times.png')
        plt.close()

    def get_average_execution_time(self):
        return np.mean([log['execution_time'] for log in self.logs])


# #### 9.2. Initialize SimpleModelMonitor

# In[66]:


model_monitor = SimpleModelMonitor()


# #### 9.3. Create a function to run an agent with monitoring
# 
# `run_agent_with_monitoring`, combines the execution of the AI sales analyst agent with performance monitoring

# **Steps to be followed:**
# 
# 1. Receive Query
#     - The function takes a query as input.
#     - This query can be any question or request related to our sales data.
# 
# 
# 2. Start Timing
#     - Record the start time just before executing the agent.
#     - This marks the beginning of the performance measurement.
# 
# 
# 3. Execute Agent
#     - Run the `agent_chain` with the given query.
#     - The `agent_chain` is the core AI component, encompassing:
#       - The language model
#       - Available tools (e.g., data analysis, plotting)
#       - Decision-making logic for using these tools
#     - It processes the query and generates a response.
# 
# 
# 4. End Timing
#     - Record the end time immediately after the agent completes its task.
# 
# 5. Calculate Execution Time
#     - Compute the total execution time by finding the difference between the start and end times.
#     - This provides the key performance metric for each interaction.
# 
# 
# 6. Log the Interaction
#     - Use `model_monitor.log_interaction(query, execution_time)` to monitor system to log
# 
# 
# 7. Return Results
# - The function returns two pieces of information:
#   - The agent's response to the query
#   - The execution time for this interaction
# 

# In[67]:


def run_agent_with_monitoring(query):
    start_time = datetime.now()
    response = agent_chain.run(input=query, chat_history="", agent_scratchpad="")
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    model_monitor.log_interaction(query, execution_time)
    return response, execution_time


# ## Step 10. Create **InsightForge, Business Intelligence Assistant** web app using Streamlit:
# 
# This Streamlit app creates a user-friendly interface for  AI-powered Business Intelligence Assistant, **InsightForge**.
# 
# It integrates data analysis, AI-powered insights, and model performance monitoring into a single, interactive web application.
# 

# ## App Structure
# 
# ### Title and Navigation
# - The app is titled **InsightForge: Business Intelligence Assistant**.
# - A sidebar navigation menu allows users to switch between different sections:
#   - Home
#   - Data Analysis
#   - AI Assistant
#   - Model Performance
# 
# ### Home Page
# - Displays a welcome message and instructions for using the app.
# 
# ### Data Analysis Page
# 1. **Sales Summary**
#    - Displays the advanced summary of sales data.
# 
# 2. **Sales Distribution by Product Category**
#    - Shows a plot of sales distribution across different product categories.
# 
# 3. **Daily Sales Trend**
#    - Presents a graph of the daily sales trend over time.
# 
# ### AI Assistant Page
# - Provides an interface for users to interact with the BI Assistant.
# - AI assistant comes with 2 different modes: `standard` and `RAG`. `Standard` directly generates a response based on the input without additional context or retrieval from external sources. Whereas `RAG` generates a response using Retrieval-Augmented Generation (RAG) along with enhanced context. The user is free to chose between the function and see how the results change. Make tweaks as needed.
# - Features:
#   - Text input for user questions
#   - Display of LLM-generated responses
#   - Execution time tracking for each query
# 
# ### Model Performance Page
# 1. **Model Evaluation**
#    - Allows users to run and view results of model evaluation.
#    - Displays:
#      - Questions
#      - Predicted answers
#      - Actual answers
#      - Correctness of predictions
#      - Overall model accuracy
# 
# 2. **Execution Time Monitoring**
#    - Shows a graph of model execution times over timestamp.
#    - Displays the average execution time.

# In[ ]:


import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


# #### 10.1 Body of the App

# In[ ]:


st.title("InsightForge: Business Intelligence Assistant :robot_face::bulb:")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "AI Assistant", "Model Performance"])


if page == "Home":
    st.write("Welcome to InsightForge, your AI-powered Business Intelligence Assistant.")
    st.write("Use the sidebar to navigate through different sections of the application.")

elif page == "Data Analysis":
    st.header("Data Analysis")

    st.subheader("Sales Summary")
    st.write(advanced_summary)

    st.subheader("Sales Distribution by Product Category")
    fig_category = plot_product_category_sales()
    st.pyplot(fig_category)

    st.subheader("Daily Sales Trend")
    fig_trend = plot_sales_trend()
    st.pyplot(fig_trend)

elif page == "AI Assistant":
    st.header("AI Sales Analyst")

    ai_mode = st.radio("Choose AI Mode:", ["Standard", "RAG Insights"])

    user_input = st.text_input("Ask a question about the sales data:")
    if user_input:
        if ai_mode == "Standard":
            start_time = datetime.now()
            response = agent_chain.run(input=user_input, chat_history="", agent_scratchpad="")
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            st.write("AI Response:")
            st.write(response)

            model_monitor.log_interaction(user_input, execution_time)
            st.write(f"Execution time: {execution_time:.2f} seconds")

        else:  # RAG Insights mode
            start_time = datetime.now()
            rag_response = generate_rag_insight(user_input)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            st.write("RAG Insight:")
            st.write(rag_response)

            model_monitor.log_interaction(user_input, execution_time)
            st.write(f"Execution time: {execution_time:.2f} seconds")

elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Model Evaluation")
    if st.button("Run Model Evaluation"):
        qa_pairs = [
        {
            "question": "What is our total sales amount?",
            "answer": f"The total sales amount is ${df['Sales'].sum():,.2f}."
        },
        {
            "question": "Which product category has the highest sales?",
            "answer": f"The product category with the highest sales is {df.groupby('Product')['Sales'].sum().idxmax()}."
        },
        {
            "question": "What is our average customer satisfaction score?",
            "answer": f"The average customer satisfaction score is {df['Customer_Satisfaction'].mean():.2f}."
        },
        # Add more question-answer pairs as needed
    ]
        eval_results = evaluate_model(qa_pairs)
        for result in eval_results:
            st.write(f"Question: {result['question']}")
            st.write(f"Predicted: {result['predicted']}")
            st.write(f"Actual: {result['actual']}")
            st.write(f"Correct: {result['correct']}")
            st.write("---")

        accuracy = sum([1 for r in eval_results if r['correct']]) / len(eval_results)
        st.write(f"Model Accuracy: {accuracy:.2%}")

    st.subheader("Execution Time Monitoring")
    fig, ax = plt.subplots()
    timestamps = [datetime.fromisoformat(log['timestamp']) for log in model_monitor.logs]
    execution_times = [log['execution_time'] for log in model_monitor.logs]
    ax.plot(timestamps, execution_times, marker='o')
    ax.set_title('Model Execution Times')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    avg_execution_time = model_monitor.get_average_execution_time()
    st.write(f"Average Execution Time: {avg_execution_time:.2f} seconds")

if __name__ == '__main__':
    pass

