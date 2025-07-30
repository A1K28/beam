package org.apache.beam.sdk.extensions.xlang;

import com.google.auto.service.AutoService;
import org.apache.beam.sdk.expansion.ExternalTransformRegistrar;
import org.apache.beam.sdk.transforms.ExternalTransformBuilder;

import java.util.Collections;
import java.util.Map;

/**
 * Registers our PrefixBuilder under its URN so that the Beam ExpansionService
 * discovers it via ServiceLoader.
 */
@AutoService(ExternalTransformRegistrar.class)
public class PrefixTransformRegistrar implements ExternalTransformRegistrar {

  @Override
  public Map<String, Class<? extends ExternalTransformBuilder<?, ?, ?>>> knownBuilders() {
    return Collections.singletonMap(PrefixBuilder.URN, PrefixBuilder.class);
  }
}
