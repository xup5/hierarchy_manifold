from diffusers import DiffusionPipeline
import torch
import os

num_images_per_prompt = 50
animal_orders = {
    "Carnivora": [
        "African lion",
        "Bengal tiger",
        "Gray wolf",
        "Domestic cat",
        "Polar bear",
        "Red fox",
        "Sea otter",
        "Raccoon",
        "Spotted hyena",
        "Giant panda"
    ],
    "Primates": [
        "Chimpanzee",
        "Bonobo",
        "Gorilla",
        "Orangutan",
        "Human",
        "Japanese macaque",
        "Howler monkey",
        "Lemur",
        "Gibbon",
        "Tarsier"
    ],
    "Lepidoptera": [
        "Monarch butterfly",
        "Swallowtail butterfly",
        "Painted lady",
        "Blue morpho butterfly",
        "Luna moth",
        "Atlas moth",
        "Cabbage white",
        "Red admiral",
        "Common buckeye",
        "Hawk moth"
    ],
    "Coleoptera": [
        "Ladybug",
        "Stag beetle",
        "Rhinoceros beetle",
        "Dung beetle",
        "Jewel beetle",
        "Firefly",
        "Weevil",
        "Ground beetle",
        "Leaf beetle",
        "Ten-lined June beetle"
    ],
    "Anura": [
        "American bullfrog",
        "Red-eyed tree frog",
        "Poison dart frog",
        "African clawed frog",
        "Wood frog",
        "Common toad",
        "Cane toad",
        "Spring peeper",
        "Glass frog",
        "Goliath frog"
    ],
    "Squamata": [
        "Komodo dragon",
        "Green iguana",
        "King cobra",
        "Garter snake",
        "Chameleon",
        "Gecko",
        "Anaconda",
        "Eastern fence lizard",
        "Viper",
        "Water monitor"
    ],
    "Diptera": [
        "Housefly",
        "Mosquito",
        "Fruit fly",
        "Horsefly",
        "Crane fly",
        "Blowfly",
        "Tsetse fly",
        "Flesh fly",
        "Sandfly",
        "Stable fly"
    ],
    "Passeriformes": [
        "House sparrow",
        "American robin",
        "Blue jay",
        "Northern cardinal",
        "European starling",
        "Barn swallow",
        "Black-capped chickadee",
        "Goldfinch",
        "Song sparrow",
        "Zebra finch"
    ],
    "Cetacea": [
        "Blue whale",
        "Humpback whale",
        "Orca",
        "Bottlenose dolphin",
        "Beluga whale",
        "Minke whale",
        "Sperm whale",
        "Narwhal",
        "Gray whale",
        "Harbor porpoise"
    ],
    "Decapoda": [
        "American lobster",
        "Red king crab",
        "Blue crab",
        "Coconut crab",
        "Pistol shrimp",
        "Ghost shrimp",
        "Atlantic shrimp",
        "Rock crab",
        "Peacock mantis shrimp",
        "Snow crab"
    ]
}

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)

pipe.to("cuda")
# pipe.enable_attention_slicing()
# pipe.enable_vae_slicing()

for order, species in animal_orders.items():
    for species in species:
        for i in range(num_images_per_prompt):
            image = pipe(
            f"A {species}",
            num_inference_steps=28,
            guidance_scale=3.5,
            ).images[0]
            # save images
            # create order and species directories if they don't exist 
            os.makedirs(f"/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/datasets/SD/animals/{order}", exist_ok=True)
            os.makedirs(f"/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/datasets/SD/animals/{order}/{species}", exist_ok=True)
            image.save(f"/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/datasets/SD/animals/{order}/{species}/{species}_{i}.png")
