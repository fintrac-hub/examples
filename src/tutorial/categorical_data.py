#  Licensed to the Fintech Open Source Foundation (FINOS) under one or
#  more contributor license agreements. See the NOTICE file distributed
#  with this work for additional information regarding copyright ownership.
#  FINOS licenses this file to you under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with the
#  License. You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import typing as _tp

import tracdap.rt.api as trac

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tracdap.rt.api import TracContext
from tracdap.rt_gen.domain.tracdap.metadata import ModelOutputSchema, ModelInputSchema, ModelParameter


def make_data():

    # Create a Pandas DataFrame with categorical column
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "fruit": pd.Categorical(["apple", "banana", "apple", "orange", "banana", "apple"]),
        "price": [1.5, 2.0, 1.0, 3.5, 2.5, 1.0],
    })

    # Convert to PyArrow Table (automatically preserves categorical as dictionary)
    table = pa.Table.from_pandas(df)

    # Write to Parquet
    pq.write_table(table, "data_with_categorical.parquet")


class CategoricalModel(trac.TracModel):

    def define_parameters(self) -> _tp.Dict[str, ModelParameter]:

        return trac.define_parameters(
            trac.P("filter_column", trac.STRING, "Filter column"),
            trac.P("filter_value", trac.STRING, "Filter value")
        )

    def define_inputs(self) -> _tp.Dict[str, ModelInputSchema]:
        return { "sample_input": trac.define_input_table(dynamic=True, label="Dynamic input")}

    def define_outputs(self) -> _tp.Dict[str, ModelOutputSchema]:
        return { "sample_output": trac.define_output_table(dynamic=True, label="Dynamic output")}

    def run_model(self, ctx: TracContext):

        filter_column = ctx.get_parameter("filter_column")
        filter_value = ctx.get_parameter("filter_value")

        sample_input = ctx.get_pandas_table("sample_input")
        filtered_output = sample_input[sample_input[filter_column] != filter_value]

        if pd.CategoricalDtype.is_dtype(filtered_output[filter_column]):
            filtered_output[filter_column].cat.remove_unused_categories()

        print(filtered_output)

        ctx.put_schema("sample_output", ctx.get_schema("sample_input"))
        ctx.put_pandas_table("sample_output", filtered_output)

if __name__ == "__main__":
    import tracdap.rt.launch as launch
    launch.launch_model(CategoricalModel, "config/categorical_data.yaml", "config/sys_config.yaml")
