package org.apache.beam.sdk.extensions.xlang;

import org.apache.beam.sdk.transforms.ExternalTransformBuilder;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.PTransform;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.TypeDescriptors;
import org.apache.beam.sdk.coders.StringUtf8Coder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class PrefixBuilder
    implements ExternalTransformBuilder<
        StringConfigurationProto.StringConfiguration.Builder,
        PCollection<? extends String>,
        PCollection<String>> {

  private static final Logger LOG = LoggerFactory.getLogger(PrefixBuilder.class);
  public static final String URN = "beam:transforms:xlang:test:prefix";

  @Override
  public PTransform<PCollection<? extends String>, PCollection<String>>
  buildExternal(StringConfigurationProto.StringConfiguration.Builder cfgBuilder) {
    String prefix = cfgBuilder.getData();
    LOG.info("[PrefixBuilder] received prefix='{}'", prefix);

    PTransform<PCollection<? extends String>, PCollection<String>> xform =
      new PTransform<PCollection<? extends String>, PCollection<String>>() {
        @Override
        public PCollection<String> expand(PCollection<? extends String> in) {
          PCollection<String> out = in.apply(
            MapElements.into(TypeDescriptors.strings())
                       .via(s -> prefix + s)
          );
          out.setCoder(StringUtf8Coder.of());
          LOG.info("[PrefixBuilder] applied MapElements + setCoder(StringUtf8Coder)");
          return out;
        }
      };

    LOG.info("[PrefixBuilder] built PTransform: {}", xform);
    return xform;
  }
}