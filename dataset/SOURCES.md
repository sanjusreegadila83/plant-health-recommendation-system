# Starter dataset sources and labels

These eight small, 330-pixel Wikimedia Commons thumbnails are included only to
let you try the upload, feature-extraction, training, and prediction workflow.
They are not a representative or scientifically balanced training set. In
particular, the healthy examples show whole tomato plants, while the disease
examples are close-up leaves. A model trained on these few images can learn
background/framing differences rather than plant health. Replace or supplement
them with many consistently framed, correctly labeled images before drawing
conclusions.

All eight files are marked CC0 on their Wikimedia Commons file pages. The
Commons links below identify the original images and their creators.

| Local file | Source image / creator | License |
| --- | --- | --- |
| `Healthy/healthy_tomato_1.jpg` | [Ripe tomato on tomato plant in home garden](https://commons.wikimedia.org/wiki/File:Ripe_tomato_on_tomato_plant_in_home_garden.jpg), Clintacc | CC0 |
| `Healthy/healthy_tomato_2.jpg` | [Tomato plant growing in a home garden](https://commons.wikimedia.org/wiki/File:Tomato_plant_growing_in_a_home_garden.jpg), Clintacc | CC0 |
| `Leaf_Disease/septoria_leaf_3006.jpg` | [Tomato septoria leaf spot 3006](https://commons.wikimedia.org/wiki/File:Tomato_septoria_leaf_spot_3006.jpg), Wolan268 | CC0 |
| `Leaf_Disease/septoria_leaf_3007.jpg` | [Tomato septoria leaf spot 3007](https://commons.wikimedia.org/wiki/File:Tomato_septoria_leaf_spot_3007.jpg), Wolan268 | CC0 |
| `Leaf_Disease/septoria_leaf_3025.jpg` | [Tomato septoria leaf spot 3025](https://commons.wikimedia.org/wiki/File:Tomato_septoria_leaf_spot_3025.jpg), Wolan268 | CC0 |
| `Leaf_Disease/septoria_leaf_3026.jpg` | [Tomato septoria leaf spot 3026](https://commons.wikimedia.org/wiki/File:Tomato_septoria_leaf_spot_3026.jpg), Wolan268 | CC0 |
| `Severe_Disease/late_blight_sporulating_lesions.jpg` | [Tomato late blight leaf sporulating lesions 1](https://commons.wikimedia.org/wiki/File:Tomato_late_blight_leaf_sporulating_lesions_1_(5816170621).jpg), Scot Nelson | CC0 |
| `Severe_Disease/late_blight_leaf_curling.jpg` | [Toomato late blight leaf curling](https://commons.wikimedia.org/wiki/File:Toomato_late_blight_leaf_curling_(5816170789).jpg), Scot Nelson | CC0 |

The `Severe_Disease` folder here is a coarse demonstration label for late blight
examples. It is not a claim that image classification can determine severity in
general. Confirm actual plant problems and treatment choices with a qualified
local agriculture expert.
