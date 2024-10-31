import pandas as pd

from src.utils import parse_args, setup_directories
from src.data import (
    remove_object_subject_overlap,
    generate_parametric_knowledge_dataset,
    generate_counter_parametric_knowledge_dataset,
    remove_entity_overlap_between_relation_groups,
)
from src.parametric_knowledge import is_parametric_object_not_in_the_prompt
from src.model import load_model_and_tokenizer
from src.prompt_model import prompt_model_and_store_activations, build_prompt_for_knowledge_probing

from src.classification.classifier import perform_classification_by_relation_group, save_classification_metrics

if __name__ == "__main__":
    # define constants
    PARAREL_PATH = "pararel_with_relation_groups.csv"
    PERMANENT_PATH = "." # change with your path
    MAX_GEN_TOKENS = 10
    SPLIT_REGEXP = r'"|\.|\n|\\n|,|\('
    SUBJ_IN_OBJ_OVERLAP_STRING_SIMILARITY_RATIO = 0.8

    #############################################
    #### 0. parse the command line arguments ####
    #############################################
    args = parse_args()
    model_name = args.model_name
    device = args.device
    nb_counter_parametric_knowledge = args.nb_counter_parametric_knowledge
    include_mlps = args.include_mlps
    include_mhsa = args.include_mhsa
    include_mlps_l1 = args.include_mlps_l1
    vertical = args.vertical
    test_on_head = args.test_on_head
    token_position = args.token_position
    n_head = args.n_head

    include_first_token = token_position == "first"
    include_object_context_token = token_position == "object"
    include_subject_query_token = token_position == "subject_query"
    include_relation_query_token = token_position == "relation_query"

    print(f"The considered token is {token_position}.")

    # setup the environment
    setup_directories(PERMANENT_PATH, model_name)

    #########################################
    #### 1. load the model and tokenizer ####
    #########################################
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    #### 2. load the reformatted ParaRel
    raw_pararel_dataset_with_statements = pd.read_csv(
        f"{PERMANENT_PATH}/paper_data/raw_pararel_with_correctly_reformatted_statements.csv",
        nrows=n_head if test_on_head else None,
    )

    relations_with_gen_desc = pd.read_excel(f"{PERMANENT_PATH}/paper_data/relation_groups_by_row_with_rel_gen_desc.xlsx")

    #######################################################################
    #### 3. build the parametric knowledge dataset for the current LLM ####
    #######################################################################
    parametric_knowledge_dataset = generate_parametric_knowledge_dataset(
        raw_pararel_dataset_with_statements,
        model,
        tokenizer,
        relations_with_gen_desc,
        device,
        MAX_GEN_TOKENS,
        split_regexp=SPLIT_REGEXP,
    )

    parametric_knowledge_dataset_without_subj_in_obj_overlap = remove_object_subject_overlap(
        parametric_knowledge_dataset, string_similarity_ratio=SUBJ_IN_OBJ_OVERLAP_STRING_SIMILARITY_RATIO
    )

    are_parametric_objects_not_in_the_prompt = [
        is_parametric_object_not_in_the_prompt(parametric_object, relations_with_gen_desc)
        for parametric_object in parametric_knowledge_dataset_without_subj_in_obj_overlap["parametric_object"]
    ]

    parametric_knowledge_dataset = parametric_knowledge_dataset_without_subj_in_obj_overlap[
        are_parametric_objects_not_in_the_prompt
    ]

    parametric_knowledge_dataset = remove_entity_overlap_between_relation_groups(parametric_knowledge_dataset)

    ###############################################################################
    #### 4. build the counter parametric knowledge dataset for the current LLM ####
    ###############################################################################
    counter_parametric_knowledge_dataset = generate_counter_parametric_knowledge_dataset(
        parametric_knowledge_dataset, nb_counter_parametric_knowledge
    )

    counter_parametric_knowledge_dataset["knowledge_probing_prompt"] = counter_parametric_knowledge_dataset.apply(
        lambda counter_knowledge: build_prompt_for_knowledge_probing(
            input_context=counter_knowledge.statement.replace(
                counter_knowledge.statement_object, counter_knowledge.counter_knowledge_object
            ),
            input_query=counter_knowledge.statement_query,
        ),
        axis=1,
    ).tolist()

    #####################################################################################################################
    #### 5. prompt the current LLM and store its activations as features and identify the knowledge source (targets) ####
    #####################################################################################################################
    prompting_results = prompt_model_and_store_activations(
        counter_parametric_knowledge_dataset=counter_parametric_knowledge_dataset,
        device=device,
        tokenizer=tokenizer,
        model_name=model_name,
        model=model,
        include_first_token=include_first_token,
        include_object_context_token=include_object_context_token,
        include_subject_query_token=include_subject_query_token,
        include_relation_query_token=include_relation_query_token,
        include_mlps=include_mlps,
        include_mlps_l1=include_mlps_l1,
        include_mhsa=include_mhsa,
        vertical=vertical,
        max_gen_tokens=MAX_GEN_TOKENS,
        split_regexp=SPLIT_REGEXP,
    )

    # save the prompting results (used to reproduce the knowledge source counting figures)
    prompting_results[["knowledge_source"]].to_pickle(f"{PERMANENT_PATH}/{model_name}_prompting_results_without_activations.pkl")

    # we don't consider ND labels in our classifier
    prompting_results = prompting_results[prompting_results.knowledge_source != "ND"]

    ####################################################
    #### 5. train the classifier on the activations ####
    ####################################################
    classification_metrics = perform_classification_by_relation_group(
        prompting_results=prompting_results,
        include_mlps=include_mlps,
        include_mlps_l1=include_mlps_l1,
        include_mhsa=include_mhsa,
        vertical=vertical,
        position=token_position,
    )

    ############################################
    #### 6. save the classification metrics ####
    ############################################
    save_classification_metrics(
        classification_metrics=classification_metrics,
        model_name=model_name,
        permanent_path=PERMANENT_PATH,
        include_mlps=include_mlps,
        include_mlps_l1=include_mlps_l1,
        include_mhsa=include_mhsa,
        vertical=vertical,
        position=token_position,
        test_on_head=test_on_head,
        n_head=n_head,
    )
