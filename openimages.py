import fiftyone as fo
import fiftyone.zoo as foz

COCO_CLASSES = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train",
    "Truck", "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter",
    "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear",
    "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase",
    "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
    "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle",
    "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut",
    "Cake", "Chair", "Couch", "Potted Plant", "Bed", "Dining Table", "Toilet",
    "TV", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave",
    "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase",
    "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"
]


# Загрузка датасета
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="validation",
    label_types=["detections"],
    classes=COCO_CLASSES,
    max_samples=5000,
    dataset_name="openimages_coco_val",
)

# Проверка поля ground_truth
sample = dataset.first()
print(sample.ground_truth)  # Должен быть типа Detections

# Экспорт в COCO формат
dataset.export(
    export_dir="openimages_exported_coco_large",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    classes=COCO_CLASSES,
)
