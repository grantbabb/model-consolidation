import os
from typing import Dict

import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.steps import TuningStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterFloat,
    ParameterInteger,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingOutput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.automl.automl import AutoMLInput
from sagemaker.automl.automlv2 import AutoMLV2, AutoMLTabularConfig
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables


def get_session(region: str) -> PipelineSession:
    return PipelineSession(boto_session=boto3.Session(region_name=region))


def get_pipeline(region: str, role: str) -> Pipeline:
    sagemaker_session = get_session(region)

    # Parameters
    feature_group_name = ParameterString(name="FeatureGroupName", default_value="churn_aws_fg-20251009-163549")
    target_column = ParameterString(name="TargetColumn", default_value="target")
    event_time_after = ParameterString(name="EventTimeAfter", default_value="")
    event_time_column = ParameterString(name="EventTimeColumn", default_value="event_time")
    train_split_ratio = ParameterFloat(name="TrainSplitRatio", default_value=0.8)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    automl_job_name = ParameterString(name="AutoMLJobName", default_value="automl-job")
    automl_s3_output = ParameterString(name="AutoMLS3Output", default_value="s3://datasets-in-out/automl-output/")
    training_output_s3 = ParameterString(name="TrainingOutputS3", default_value="s3://datasets-in-out/train-output/")
    max_parallel_jobs = ParameterInteger(name="MaxParallelJobs", default_value=5)

    # Step 1: Feature Store read via Processing
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sagemaker_session,
    )

    base_bucket = sagemaker_session.default_bucket()
    base_prefix = Join(on="/", values=["pipelines", ExecutionVariables.PIPELINE_EXECUTION_ID])
    processed_train_s3 = Join(on="/", values=["s3:/", base_bucket, base_prefix, "processed", "train"])  # s3://bucket/prefix/processed/train
    processed_val_s3 = Join(on="/", values=["s3:/", base_bucket, base_prefix, "processed", "validation"])  # s3://bucket/prefix/processed/validation
    athena_results_s3 = Join(on="/", values=["s3:/", base_bucket, base_prefix, "athena-results"])  # s3://bucket/prefix/athena-results

    processing_step = sklearn_processor.run(
        code=os.path.join(os.path.dirname(__file__), "fs_read", "feature_store_extract.py"),
        arguments=[
            "--feature-group-name", feature_group_name,
            "--output-train-dir", "/opt/ml/processing/output/train",
            "--output-validation-dir", "/opt/ml/processing/output/validation",
            "--target-column", target_column,
            "--event-time-after", event_time_after,
            "--event-time-column", event_time_column,
            "--train-split-ratio", train_split_ratio,
            "--athena-output-s3", athena_results_s3,
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output/train", output_name="train", destination=processed_train_s3),
            ProcessingOutput(source="/opt/ml/processing/output/validation", output_name="validation", destination=processed_val_s3),
        ],
    )

    # Step 2: AutoML to get best candidate
#    automl = AutoML(
#        role=role,
#        target_attribute_name=target_column,
#        output_path=automl_s3_output,
#        sagemaker_session=sagemaker_session,
#        problem_type="BinaryClassification",
#        job_objective=AutoMLJobObjective(metric_name="AUC"),
#        max_candidates=10,
#    )

    problem_config = AutoMLTabularConfig(
        target_attribute_name='user_ churned',
        problem_type='BinaryClassification', # Or 'Regression', 'BinaryClassification'
        
    )

    automl = AutoMLV2(
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name='my-automl-job',
        problem_config=problem_config,
        output_path=automl_s3_output,
        job_objective={'MetricName': 'Accuracy'}
    )

    automl_input = AutoMLInput(
        inputs=processed_train_s3,
        target_attribute_name=target_column,
    )

    step_args = automl.fit(inputs=automl_input, job_name='my-automl-job')
    automl_step = AutoMLStep(name="AutoModel", step_args=step_args)

#    automl_step = AutoMLStep(
#        name="AutoModel",
#        automl=automl,
#        inputs=automl_input,
#        job_name=automl_job_name,
#    )

    # Step 3: Hyperparameter Tuning (HPO) with XGBoost
    xgb_image_uri = image_uris.retrieve(framework="xgboost", region=region, version="1.5-1")

    estimator = Estimator(
        image_uri=xgb_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=training_output_s3,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        num_round=200,
    )

    hyperparameter_ranges = {
        "eta": ContinuousParameter(0.01, 0.5),
        "max_depth": IntegerParameter(3, 10),
        "min_child_weight": ContinuousParameter(1.0, 10.0),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:auc",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=20,
        max_parallel_jobs=max_parallel_jobs,
        objective_type="Maximize",
    )

    tuning_step = TuningStep(
        name="HPO",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=processed_train_s3,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=processed_val_s3,
                content_type="text/csv",
            ),
        },
    )

    # Step 4: Register best model from HPO
    model_package_group = ParameterString(name="ModelPackageGroupName", default_value="MyModelPackageGroup")

    register_step = RegisterModel(
        name="RegisterBestModel",
        estimator=estimator,
        model_data=tuning_step.get_top_model_s3_uri(top_k=0),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="PendingManualApproval",
    )

    pipeline = Pipeline(
        name="FS-AutoML-HPO-Register",
        parameters=[
            feature_group_name,
            target_column,
            event_time_after,
            event_time_column,
            train_split_ratio,
            processing_instance_type,
            processing_instance_count,
            automl_job_name,
            automl_s3_output,
            training_output_s3,
            max_parallel_jobs,
            model_package_group,
        ],
        steps=[processing_step, automl_step, tuning_step, register_step],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


def pipeline_definition(region: str, role: str) -> Dict:
    return get_pipeline(region, role).definition()

pipeline_definition('us-west-2', 'arn:aws:iam::063299843915:role/service-role/AmazonSageMaker-ExecutionRole-20250522T112887')