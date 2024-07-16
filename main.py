# LLM App to extract key data from a product review

import streamlit as st
from enum import Enum
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate



CREATIVITY=0
TEMPLATE = """\
For the following text, extract the following \
information:

sentiment: Is the customer happy with the product?
Answer Positive if yes, Negative if \
not, Neutral if either of them, or Unknown if unknown.

delivery_days: How many days did it take \
for the product to arrive? If this \
information is not found, output No information about this.

price_perception: How does it feel the customer about the price?
Anwer Expensive if the customer feels the product is expensive,
Cheap if the customer feels the product is cheap,
not, Neutral if either of them, or Unknown if unknown.

Format the output as bullet-points text with the \
following keys:
- Sentiment
- How long took it to deliver?
- How was the price perceived?

Input example:
This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.

Output example:
- Sentiment: Positive
- How long took it to deliver? 2 days
- How was the price perceived? Cheap

text: {review}
"""


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


# Defining prompt template
class FinalPromptTemplate:
    def __init__(self, review:str) -> None:
        self.review=review
        
    def generate(self) -> str:
        prompt = PromptTemplate(
            input_variables=["review"],
            template=TEMPLATE
        )
        final_prompt = prompt.format(
            review=self.review
        )

        return final_prompt


class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e
        

class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def validate_api_key(self, key:str):
        if not key:
            st.sidebar.warning("Please enter your API Key")
            # st.stop()
        else:    
            if (key.startswith("sk-") or key.startswith("gsk_")):
                st.sidebar.success("Received valid API Key!")
            else:
                st.sidebar.error("Invalid API Key!")

    def get_api_key(self):
        
        # Get the API Key to query the model
        input_text = st.sidebar.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )

        # Validate the API key
        self.validate_api_key(input_text)
        return input_text
    
    def create(self):
        try:
            # Set the page title for blog post
            st.set_page_config(page_title="Extract Key Information from Product Reviews")
            st.markdown("<h1 style='text-align: center;'>Extract Key Information from Product Reviews</h1>", unsafe_allow_html=True)
            st.markdown("Extracting following key information from a product review:")
            st.markdown("""
                - Sentiment
                - How long took it to deliver?
                - How was its price perceived?
                """)

            # Select the model provider
            option_model_provider = st.sidebar.selectbox(
                    'Model Provider',
                    ('GroqCloud', 'OpenAI')
                )
            
            # Input API Key for model to query
            api_key = self.get_api_key()

            # Input
            st.markdown("### Enter the product review")
            review_input = st.text_area(label="Product Review", label_visibility="collapsed", placeholder="Your Product Review...", key="review_input")
            if len(review_input.split(" ")) > 7000:
                st.write("Please enter a shorter product review. The maximum length is 700 words.")
                st.stop()

            # Output
            st.markdown("### Key Data Extracted:")
            if review_input:
                if not api_key:
                    st.warning("Please insert your API Key", icon="⚠️")
                    st.stop()

                # Generate the final prompt
                final_prompt = FinalPromptTemplate(review_input)
                print("Final Prompt: ", final_prompt.generate())
                
                # Load the LLM model
                llm_model = LLMModel(model_provider=option_model_provider)
                llm = llm_model.load(api_key=api_key)

                key_data_extraction = llm.invoke(final_prompt.generate())
                st.write(key_data_extraction.content)

        except Exception as e:
            import traceback
            st.error(str(e), icon=":material/error:")
            # st.error(traceback.format_exc())


def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()