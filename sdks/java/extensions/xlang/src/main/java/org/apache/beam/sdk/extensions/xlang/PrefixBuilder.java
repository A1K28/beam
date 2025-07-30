package org.apache.beam.sdk.extensions.xlang;

import org.apache.beam.sdk.transforms.ExternalTransformBuilder;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.PTransform;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.TypeDescriptors;

/**
 * Now parameterized on the mutable Builder, not the immutable message.
 */
public class PrefixBuilder
    implements ExternalTransformBuilder<
        StringConfigurationProto.StringConfiguration.Builder,
        PCollection<? extends String>,
        PCollection<String>> {

  public static final String URN = "beam:transforms:xlang:test:prefix";

  @Override
  public PTransform<PCollection<? extends String>, PCollection<String>>
  buildExternal(StringConfigurationProto.StringConfiguration.Builder cfgBuilder) {
    // cfgBuilder.getData() is already populated via setter calls under the hood
    String prefix = cfgBuilder.getData();
    return MapElements
        .into(TypeDescriptors.strings())
        .via((String s) -> prefix + s);
  }
}
