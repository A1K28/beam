package org.apache.beam.sdk.extensions.xlang;

import org.apache.beam.sdk.transforms.ExternalTransformBuilder;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.PTransform;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.TypeDescriptors;
import org.apache.beam.sdk.coders.StringUtf8Coder;


public class PrefixBuilder
    implements ExternalTransformBuilder<
        StringConfigurationProto.StringConfiguration.Builder,
        PCollection<? extends String>,
        PCollection<String>> {

  public static final String URN = "beam:transforms:xlang:test:prefix";

  @Override
  public PTransform<PCollection<? extends String>, PCollection<String>>
  buildExternal(StringConfigurationProto.StringConfiguration.Builder cfgBuilder) {
    final String prefix = cfgBuilder.getData();

    // Wrap in an anonymous PTransform so we can explicitly setCoder(...)
    return new PTransform<PCollection<? extends String>, PCollection<String>>() {
      @Override
      public PCollection<String> expand(PCollection<? extends String> input) {
        PCollection<String> out = input.apply(
            MapElements
              .into(TypeDescriptors.strings())
              .via((String x) -> prefix + x)
        );
        // this is what ensures the expansion response uses beam:coder:string_utf8:v1
        out.setCoder(StringUtf8Coder.of());
        return out;
      }
    };
  }
}
