from openai import AzureOpenAI
# test_llm_config = LLMConfig(
#     api_key=",
#     api_type=,
#     api_version=,
#     base_url="",
#
# )
# gets the API Key from environment variable AZURE_OPENAI_API_KEY

if __name__ == '__main__':
    client = AzureOpenAI(
        api_key="",
        api_version="",
        base_url="",

    )

    completion = client.chat.completions.create(
        model="deployment-name",  # e.g. gpt-35-instant
        messages=[
            {
                "role": "user",
                "content": "How do I output all files in a directory using Python?",
            },
        ],
    )
    print(completion.model_dump_json(indent=2))
