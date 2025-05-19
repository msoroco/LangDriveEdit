import random
import logging
import torch
from torch.utils.data import DataLoader

# Setup logger
logger = logging.getLogger(__name__)


class ResilientDataLoader(torch.utils.data.DataLoader):
    """DataLoader that replaces problematic samples on-the-fly"""
    
    def __init__(self, *args, **kwargs):
        self.replacement_dataset = kwargs.pop('replacement_dataset', None)
        if self.replacement_dataset is None:
            self.replacement_dataset = args[0]  # Use the original dataset for replacements
        self.max_retries = kwargs.pop('max_retries', 3)
        super().__init__(*args, **kwargs)
    
    def __iter__(self):
        # Create a pool of replacement indices
        self.replacement_indices = list(range(len(self.replacement_dataset)))
        random.shuffle(self.replacement_indices)
        self.replacement_index = 0
        
        iterator = super().__iter__()
        return ResilientBatchIterator(iterator, self)


class ResilientBatchIterator:
    """Iterator that handles problematic samples by replacing them"""
    
    def __init__(self, dataloader_iter, dataloader):
        self.dataloader_iter = dataloader_iter
        self.dataloader = dataloader
        self.max_retries = dataloader.max_retries
    
    def __iter__(self):
        return self
    
    def __next__(self):
        for retry in range(self.max_retries):
            try:
                batch = next(self.dataloader_iter)
                return batch
            except StopIteration:
                raise
            except Exception as e:
                if retry == self.max_retries - 1:
                    # On last retry, try to create a completely new batch from replacement samples
                    logger.warning(f"Batch failed after {retry+1} retries. Creating replacement batch. Error: {str(e)}")
                    return self._create_replacement_batch()
                else:
                    logger.info(f"Batch loading error (retry {retry+1}/{self.max_retries}): {str(e)}")
    
    def _create_replacement_batch(self):
        """Create a replacement batch from the replacement dataset"""
        batch_size = self.dataloader.batch_size
        replacement_samples = []
        
        # Get replacement samples
        for _ in range(batch_size):
            # Cycle through replacement indices if we run out
            if self.dataloader.replacement_index >= len(self.dataloader.replacement_indices):
                random.shuffle(self.dataloader.replacement_indices)
                self.dataloader.replacement_index = 0
                
            idx = self.dataloader.replacement_indices[self.dataloader.replacement_index]
            self.dataloader.replacement_index += 1
            
            try:
                sample = self.dataloader.replacement_dataset[idx]
                replacement_samples.append(sample)
            except Exception:
                # If this sample also fails, skip it
                continue
                
        if not replacement_samples:
            raise RuntimeError("Failed to create replacement batch - all samples failed")
            
        # Use the dataloader's collate_fn to create a proper batch
        return self.dataloader.collate_fn(replacement_samples)
    

class ResilientDatasetWrapper:
    """A wrapper for datasets that handles corrupted images"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Try the original index first
        try:
            return self.dataset[idx]
        except Exception as e:
            logger.warning(f"Error loading sample at index {idx}: {str(e)}")
            
            # Try up to 15 random indices
            for _ in range(15):
                try:
                    random_idx = random.randint(0, len(self.dataset) - 1)
                    return self.dataset[random_idx]
                except Exception:
                    print(f"Failed to load sample at random index {random_idx}, trying again...")
                    continue
                    
            # If all attempts fail, raise an error
            print(f"index: {idx}, data: {self.dataset[idx]}")
            raise RuntimeError(f"Failed to load sample at index {idx} and all random attempts")
            

import gc
from PIL import Image
import numpy as np

class ResilientDatasetWrapperGC:
    """A wrapper for datasets that handles corrupted images with immediate cleanup"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self._cleanup_frequency = 16
        self._last_cleanup = 0
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def cleanup_pil_images(self, sample):
        """Clean up PIL images in a sample"""
        if isinstance(sample, dict):
            for key, value in sample.items():
                if isinstance(value, Image.Image):
                    value.close()
                    del value
    
    def __getitem__(self, idx):
        # Periodic cleanup
        if idx - self._last_cleanup >= self._cleanup_frequency:
            self._last_cleanup = idx
            gc.collect()
            # torch.cuda.empty_cache()
        
        try:
            # Get sample and immediately convert PIL to numpy
            current_sample = self.dataset[idx]
            processed_sample = {}
            
            for key, value in current_sample.items():
                if isinstance(value, Image.Image):
                    # Convert to numpy and close PIL image immediately
                    processed_sample[key] = np.array(value)
                    value.close()
                    del value
                else:
                    processed_sample[key] = value
            # Clear references
            del current_sample

            if self.transform is not None:
                processed_sample = self.transform(processed_sample)
            return processed_sample
            
        except Exception as e:
            logger.warning(f"Error loading sample at index {idx}: {str(e)}")
            
            # Try up to 15 random indices
            for _ in range(15):
                try:
                    random_idx = random.randint(0, len(self.dataset) - 1)
                    sample = self.dataset[random_idx]
                    processed_sample = {}
                    
                    for key, value in sample.items():
                        if isinstance(value, Image.Image):
                            processed_sample[key] = np.array(value)
                            value.close()
                            del value
                        else:
                            processed_sample[key] = value     
                    del sample
                    
                    if self.transform is not None:
                        processed_sample = self.transform(processed_sample)
                    return processed_sample
                    
                except Exception as e:
                    continue
            
            raise RuntimeError(f"Failed to load sample at index {idx}")


from torch.utils.data import IterableDataset

class IterableWithTransform(IterableDataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for example in self.dataset:
            yield self.transform(example)