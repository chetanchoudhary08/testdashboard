import os
import streamlit as st
import pyodbc
import faiss
import numpy as np
import openai

##############################
# 1) AZURE OPENAI CONFIG
##############################
openai.api_type = "azure"
openai.api_base = "https://cog-dfecc4yseotvi.openai.azure.com"
openai.api_key = "59b98ee2a85468d8611749e24eac5fe"
openai.api_version = "2023-05-15"

EMBED_MODEL_NAME = "embedding"
CHAT_MODEL_NAME = "gpt-4o"

##############################
# 2) PRICING & EXCHANGE RATES
##############################
# Example: GPT-4o
GPT4_PROMPT_COST_PER_1K_TOKENS = 0.0025   # $2.50 per million tokens
GPT4_COMPLETION_COST_PER_1K_TOKENS = 0.01 # $10.00 per million tokens

# Example: Ada v2 embedding
EMBED_COST_PER_1K_TOKENS = 0.0001  # $0.10 per million tokens

# 1 USD = 86 INR, 1 USD = 0.81 GBP (example values)
USD_TO_INR = 86.0
USD_TO_GBP = 0.81

##############################
# 3) SESSION-STATE USAGE
##############################
if "usage" not in st.session_state:
    st.session_state["usage"] = {
        "embedding_tokens": 0,
        "embedding_cost": 0.0,
        "chat_prompt_tokens": 0,
        "chat_prompt_cost": 0.0,
        "chat_completion_tokens": 0,
        "chat_completion_cost": 0.0,
        "chat_cost": 0.0
    }

def get_usage_snapshot():
    return dict(st.session_state["usage"])

def usage_diff(before, after):
    return {
        "embedding_tokens": after["embedding_tokens"] - before["embedding_tokens"],
        "embedding_cost": after["embedding_cost"] - before["embedding_cost"],
        "chat_prompt_tokens": after["chat_prompt_tokens"] - before["chat_prompt_tokens"],
        "chat_prompt_cost": after["chat_prompt_cost"] - before["chat_prompt_cost"],
        "chat_completion_tokens": after["chat_completion_tokens"] - before["chat_completion_tokens"],
        "chat_completion_cost": after["chat_completion_cost"] - before["chat_completion_cost"],
        "chat_cost": after["chat_cost"] - before["chat_cost"]
    }

##############################
# 4) MULTI-CURRENCY COST DISPLAY
##############################
def format_cost_multi_currency(usd_amount):
    """
    Show cost in USD, INR, and GBP.
    Precision: 
      - USD: 6 decimals
      - INR: 2 decimals
      - GBP: 4 decimals
    """
    inr_amount = usd_amount * USD_TO_INR
    gbp_amount = usd_amount * USD_TO_GBP
    return (
        f"($ {usd_amount:.6f} | "
        f"₹ {inr_amount:.2f} | "
        f"£ {gbp_amount:.4f})"
    )

def display_usage_diff(diff, label="Operation"):
    if any(value != 0 for key, value in diff.items() if "tokens" in key):
        st.markdown(f"**Cost for {label}:**")

        # Embedding
        if diff["embedding_tokens"]:
            st.write(
                f"- Embedding tokens: {diff['embedding_tokens']} "
                f"(cost: {format_cost_multi_currency(diff['embedding_cost'])})"
            )

        # Chat
        cp_toks = diff["chat_prompt_tokens"]
        cc_toks = diff["chat_completion_tokens"]
        if cp_toks or cc_toks:
            if cp_toks:
                st.write(
                    f"- Chat prompt tokens: {cp_toks} "
                    f"(cost: {format_cost_multi_currency(diff['chat_prompt_cost'])})"
                )
            if cc_toks:
                st.write(
                    f"- Chat completion tokens: {cc_toks} "
                    f"(cost: {format_cost_multi_currency(diff['chat_completion_cost'])})"
                )
            if diff["chat_cost"]:
                st.write(f"- **Total chat cost**: {format_cost_multi_currency(diff['chat_cost'])}")

        total_usd = diff["embedding_cost"] + diff["chat_cost"]
        st.markdown(
            f"**Total cost for this {label}**: {format_cost_multi_currency(total_usd)}"
        )
    else:
        st.write(f"No usage info returned for {label}.")

##############################
# 5) USAGE UPDATE FUNCTIONS
##############################
def update_embedding_usage(response):
    usage = response.get("usage")
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        st.session_state["usage"]["embedding_tokens"] += prompt_tokens
        cost = (prompt_tokens / 1000.0) * EMBED_COST_PER_1K_TOKENS
        st.session_state["usage"]["embedding_cost"] += cost

def update_chat_usage(response):
    usage = response.get("usage")
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        st.session_state["usage"]["chat_prompt_tokens"] += prompt_tokens
        st.session_state["usage"]["chat_completion_tokens"] += completion_tokens

        prompt_cost = (prompt_tokens / 1000.0) * GPT4_PROMPT_COST_PER_1K_TOKENS
        completion_cost = (completion_tokens / 1000.0) * GPT4_COMPLETION_COST_PER_1K_TOKENS

        st.session_state["usage"]["chat_prompt_cost"] += prompt_cost
        st.session_state["usage"]["chat_completion_cost"] += completion_cost
        st.session_state["usage"]["chat_cost"] += (prompt_cost + completion_cost)

##############################
# 6) FAISS SETUP
##############################
if "faiss_index" not in st.session_state:
    D = 1536
    index = faiss.IndexFlatL2(D)
    st.session_state["faiss_index"] = index
    st.session_state["documents"] = []

##############################
# 7) FETCH SCHEMA WITH RELATIONSHIPS
##############################
def get_db_schema(connection_string):
    """
    Returns a list of dict: [
      {
        "table_name": str,
        "columns": [
           {"column_name": str, "data_type": str}
        ],
        "relationships": [
           {
             "constraint_name": ...,
             "fk_column": ...,
             "pk_table": ...,
             "pk_column": ...,
             "relationship_desc": ""
           }
        ]
      },
      ...
    ]
    """
    tables_info = []
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # 1) List base tables
        table_query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE='BASE TABLE'
        ORDER BY TABLE_NAME
        """
        cursor.execute(table_query)
        table_names = [row[0] for row in cursor.fetchall()]

        # 2) For each table, get columns
        for tbl in table_names:
            col_query = f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{tbl}'
            ORDER BY ORDINAL_POSITION
            """
            cursor.execute(col_query)
            columns = []
            for col_row in cursor.fetchall():
                columns.append({
                    "column_name": col_row[0],
                    "data_type": col_row[1]
                })

            tables_info.append({
                "table_name": tbl,
                "columns": columns,
                "relationships": []
            })

        # 3) FK relationships
        rel_query = """
        SELECT
            FK.TABLE_NAME AS FK_Table,
            FK_COL.COLUMN_NAME AS FK_Column,
            PK.TABLE_NAME AS PK_Table,
            PK_COL.COLUMN_NAME AS PK_Column,
            RC.CONSTRAINT_NAME AS Constraint_Name
        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS RC
        INNER JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS FK
            ON RC.CONSTRAINT_NAME = FK.CONSTRAINT_NAME
        INNER JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS PK
            ON RC.UNIQUE_CONSTRAINT_NAME = PK.CONSTRAINT_NAME
        INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE FK_COL
            ON RC.CONSTRAINT_NAME = FK_COL.CONSTRAINT_NAME
        INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE PK_COL
            ON PK.CONSTRAINT_NAME = PK_COL.CONSTRAINT_NAME
        """
        cursor.execute(rel_query)
        rel_rows = cursor.fetchall()

        relationship_map = {}
        for fk_table, fk_col, pk_table, pk_col, constraint_name in rel_rows:
            if fk_table not in relationship_map:
                relationship_map[fk_table] = []
            relationship_map[fk_table].append({
                "constraint_name": constraint_name,
                "fk_column": fk_col,
                "pk_table": pk_table,
                "pk_column": pk_col,
                "relationship_desc": ""
            })

        # 4) Attach relationships
        for tbl_dict in tables_info:
            tbl = tbl_dict["table_name"]
            if tbl in relationship_map:
                tbl_dict["relationships"] = relationship_map[tbl]

        conn.close()
    except Exception as e:
        st.error(f"Error connecting/fetching schema: {e}")

    return tables_info

##############################
# 8) EMBEDDING FUNCTION
##############################
def embed_text_azure(text):
    try:
        response = openai.Embedding.create(
            input=text,
            engine=EMBED_MODEL_NAME
        )
        update_embedding_usage(response)
        embedding = response["data"][0]["embedding"]
        return np.array(embedding, dtype='float32')
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

##############################
# 9) CHAT COMPLETION
##############################
def generate_sql_with_azure_openai(system_prompt, user_message):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    try:
        completion = openai.ChatCompletion.create(
            engine=CHAT_MODEL_NAME,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
        update_chat_usage(completion)
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error calling Azure ChatCompletion: {e}")
        return "Error generating SQL"

##############################
# 10) BUILD & STORE EMBEDDINGS (WITH RELATIONSHIPS)
##############################
def build_and_store_embeddings():
    st.session_state["faiss_index"].reset()
    st.session_state["documents"].clear()

    schema_info = st.session_state["schema_info"]
    index = st.session_state["faiss_index"]

    vectors = []
    for i, table_dict in enumerate(schema_info):
        tbl_name = table_dict["table_name"]
        tbl_desc = table_dict.get("table_description", "")

        # Build columns text
        col_lines = []
        for c in table_dict["columns"]:
            col_lines.append(
                f"- {c['column_name']} (Data type: {c['data_type']}, Desc: {c.get('description','')})"
            )
        col_text = "\n".join(col_lines)

        # Build relationships text
        relationship_lines = []
        for r in table_dict["relationships"]:
            relationship_lines.append(
                f"- {r['constraint_name']}: FK {r['fk_column']} references "
                f"{r['pk_table']}.{r['pk_column']} "
                f"(Desc: {r.get('relationship_desc','')})"
            )
        if relationship_lines:
            relationships_text = "\n".join(relationship_lines)
        else:
            relationships_text = "None"

        doc_text = f"""
TABLE NAME: {tbl_name}
DESCRIPTION: {tbl_desc}

COLUMNS:
{col_text}

RELATIONSHIPS:
{relationships_text}
""".strip()

        vector = embed_text_azure(doc_text)
        if vector is not None:
            vectors.append(vector)
            st.session_state["documents"].append({
                "id": i,
                "text": doc_text,
                "table_name": tbl_name
            })

    if vectors:
        vectors_np = np.array(vectors, dtype='float32')
        index.add(vectors_np)
        st.success(f"Stored {len(vectors)} table documents in FAISS.")
    else:
        st.warning("No vectors were generated.")

##############################
# 11) SEARCH & GENERATE SQL
##############################
def run_query(user_query):
    query_vec = embed_text_azure(user_query)
    if query_vec is None:
        return "Embedding failed. Cannot proceed.", []

    # top_k = 3
    index = st.session_state["faiss_index"]
    top_k = 3
    D, I = index.search(np.array([query_vec], dtype='float32'), top_k)

    retrieved_texts = []
    for idx in I[0]:
        if idx == -1:
            continue
        doc = next((d for d in st.session_state["documents"] if d["id"] == idx), None)
        if doc:
            retrieved_texts.append(doc["text"])

    # Build system prompt
    combined_context = "\n\n".join(retrieved_texts)
    system_prompt = f"""
You are an AI assistant that writes SQL for Microsoft SQL Server. 
You have knowledge of only the following schema information:

{combined_context}

Follow these rules:
1. Only use the tables/columns in the above schema.
2. If something is missing, make your best guess.
"""

    sql_answer = generate_sql_with_azure_openai(system_prompt, user_query)
    return sql_answer, retrieved_texts

##############################
# 12) STREAMLIT UI
##############################
def show_schema_ui():
    st.header("1. DB Connection & Schema Extraction")

    conn_str_user_input = st.text_input(
        "MS SQL Server Connection String",
        help="e.g. Driver={ODBC Driver 18 for SQL Server};Server=xxx;Database=xxx;UID=xxx;PWD=xxx;"
    )

    # [Demo override]
    ms1server = 'mssqlrag.database.windows.net'
    ms1database = 'testdb'
    ms1username = 'chetan'
    ms1password = 'Password1'
    conn_str_user_input = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={ms1server};DATABASE={ms1database};UID={ms1username};PWD={ms1password}'

    if st.button("Fetch Schema"):
        if not conn_str_user_input:
            st.error("Please enter a valid connection string.")
            return
        with st.spinner("Fetching schema..."):
            st.session_state["schema_info"] = get_db_schema(conn_str_user_input)
        if st.session_state["schema_info"]:
            st.success("Schema fetched successfully!")

    # If schema_info is present, let user edit table metadata
    if "schema_info" in st.session_state and st.session_state["schema_info"]:
        st.subheader("2. Edit Table Metadata")
        for table_dict in st.session_state["schema_info"]:
            tbl_name = table_dict["table_name"]
            with st.expander(f"Table: {tbl_name}", expanded=False):
                # Table desc
                table_desc_key = f"desc_{tbl_name}"
                existing_val = st.session_state.get(table_desc_key, f"Description for {tbl_name}...")
                table_desc = st.text_area(
                    f"Description for {tbl_name}",
                    key=table_desc_key,
                    value=existing_val
                )

                # Column desc
                col_contexts = []
                for col in table_dict["columns"]:
                    col_key = f"{tbl_name}_{col['column_name']}"
                    default_txt = f"{col['column_name']} (Data type: {col['data_type']})"
                    user_txt = st.text_input(
                        f"{tbl_name}.{col['column_name']} details",
                        key=col_key,
                        value=default_txt
                    )
                    col_contexts.append({
                        "column_name": col["column_name"],
                        "data_type": col["data_type"],
                        "description": user_txt
                    })

                # Relationship desc
                if table_dict["relationships"]:
                    st.markdown("#### Relationships:")
                    for r in table_dict["relationships"]:
                        rel_key = f"rel_desc_{tbl_name}_{r['constraint_name']}"
                        rel_default = st.session_state.get(
                            rel_key,
                            f"Description for {r['constraint_name']} ..."
                        )
                        rel_label = (
                            f"FK: {r['fk_column']} → "
                            f"PK: {r['pk_table']}.{r['pk_column']} "
                            f"(Constraint: {r['constraint_name']})"
                        )
                        rel_desc = st.text_input(rel_label, key=rel_key, value=rel_default)
                        r["relationship_desc"] = rel_desc

                # Update the table_dict with user inputs
                table_dict["table_description"] = table_desc
                table_dict["columns"] = col_contexts

        # Generate embeddings
        if st.button("Generate & Store Embeddings"):
            before_usage = get_usage_snapshot()
            build_and_store_embeddings()
            after_usage = get_usage_snapshot()
            diff = usage_diff(before_usage, after_usage)
            display_usage_diff(diff, label="Schema Embeddings")

            # Optionally show final embedded docs
            if st.session_state["documents"]:
                with st.expander("Show Embedded Table Docs"):
                    for doc in st.session_state["documents"]:
                        st.write(f"**Doc ID**: {doc['id']} | **Table**: {doc['table_name']}")
                        st.code(doc["text"], language="markdown")


def show_query_ui():
    st.header("3. Query with Azure OpenAI")

    user_query = st.text_area("Enter your request (e.g. 'Show me total order amount by user'):")

    if st.button("Ask"):
        if not st.session_state["documents"]:
            st.error("No table documents stored. Please generate embeddings first.")
            return

        with st.spinner("Generating SQL..."):
            before_usage = get_usage_snapshot()
            sql_result, retrieved_texts = run_query(user_query)
            after_usage = get_usage_snapshot()
            diff = usage_diff(before_usage, after_usage)

        # Show retrieved chunks
        st.subheader("Retrieved Schema Chunks (Top Matches):")
        for i, chunk in enumerate(retrieved_texts, start=1):
            with st.expander(f"Chunk {i}"):
                st.code(chunk, language="markdown")

        # Show generated SQL
        st.subheader("Generated SQL:")
        st.code(sql_result, language="sql")

        # Show cost for this query
        display_usage_diff(diff, label="Query")

        # Attempt to scroll to bottom
        # scroll_script = """
        #     <script>
        #     window.scrollTo(0, document.body.scrollHeight);
        #     </script>
        # """
        # st.markdown(scroll_script, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="RAG SQL Demo w/ Relationships", layout="wide")
    st.title("RAG for Large SQL Server Schema (Azure OpenAI + FAISS)")

    show_schema_ui()
    st.markdown("---")
    show_query_ui()

    # Overall usage summary
    usage = st.session_state["usage"]
    total_cost_usd = usage["embedding_cost"] + usage["chat_cost"]

    st.markdown("---")
    st.subheader("Overall Cumulative Usage")
    st.write(
        f"**Embedding tokens**: {usage['embedding_tokens']} "
        f"(cost: {format_cost_multi_currency(usage['embedding_cost'])})"
    )
    st.write(
        f"**Chat prompt tokens**: {usage['chat_prompt_tokens']} "
        f"(cost: {format_cost_multi_currency(usage['chat_prompt_cost'])})"
    )
    st.write(
        f"**Chat completion tokens**: {usage['chat_completion_tokens']} "
        f"(cost: {format_cost_multi_currency(usage['chat_completion_cost'])})"
    )
    st.write(
        f"**Total chat cost**: {format_cost_multi_currency(usage['chat_cost'])}"
    )
    st.write(
        f"**Overall total cost**: {format_cost_multi_currency(total_cost_usd)}"
    )

if __name__ == "__main__":
    main()
