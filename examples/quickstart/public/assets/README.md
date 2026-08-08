# Collider Lab sample assets

These samples are bundled so the demo works offline and its behavior does not depend on a third-party server. Every upstream revision and source hash is pinned.

| Bundled file | Upstream file | Changes | Source SHA-256 | Output bytes / SHA-256 |
| --- | --- | --- | --- | --- |
| `clearcoat-wicker.glb` | Khronos [`ClearcoatWicker.glb`](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/ClearcoatWicker/glTF-Binary/ClearcoatWicker.glb) | Active-scene triangle geometry flattened; transforms applied; render-only attributes, materials, and embedded images removed | `f162b0cd7f8e6b7cef211eec57762165a78039676b8592ce1f965e2ddb34e843` | 58,304 / `33f5672b053a64809e0f368d37e4ddc4ccf24d247e8aeb93f4b49c795a0be81d` |
| `iridescent-dish-with-olives.glb` | Khronos [`IridescentDishWithOlives.glb`](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/IridescentDishWithOlives/glTF-Binary/IridescentDishWithOlives.glb) | Active-scene triangle geometry flattened; transforms applied; render-only attributes, materials, and embedded images removed | `1540b4a36b790a907f4824cfe848ba481b3da3cc8070172b7b3ba178f78a7ed1` | 472,448 / `b4e1f52da6edd8649610b9bb1c021bdb3e5a327642ef5fdb50bbfad8638cfbc3` |
| `barramundi-fish.glb` | Khronos [`BarramundiFish.glb`](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf/Models/BarramundiFish/glTF-Binary/BarramundiFish.glb) | Active-scene triangle geometry flattened; transforms applied; render-only attributes, materials, and embedded images removed | `ecc3bafb6b00f2c8b810863c388e3768a7b7ea0d0335e8cb8c574c266e571f4a` | 73,328 / `d21b3fde5c18f075c41225b582a75e64dadaef40d1087a11fb59dec63175fd41` |

Clearcoat Wicker is published by Eric Chadwick under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Iridescent Dish with Olives is published by Eric Chadwick and Wayfair, LLC under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution is provided here and in the pinned upstream model metadata. Barramundi Fish is published by Microsoft for Everything under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). All three are distributed through Khronos' [glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repository.

Clearcoat Wicker's visible weave is material-driven; its geometry-only sample is a dense smooth sphere. It is included to demonstrate aggressive convex-surface simplification, not woven geometry.

Run `npm run prepare:samples` to reproduce the bundled geometry-only GLBs. The script pins and verifies each upstream source hash before writing an output.
