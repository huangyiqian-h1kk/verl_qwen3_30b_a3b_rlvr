"""Preserve project prompts, tools, source metadata and budget fingerprints."""
from multidomain.dataset import MultiDomainDataset


class MOPDDataset(MultiDomainDataset):
    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.qualification = config.get("mopd_qualification", False)

    def __getitem__(self, index):
        value = super().__getitem__(index)
        row = self.rows[index]
        value["agent_name"] = "d4_plain_mopd"
        value["mopd_split"] = row["mopd_split"]
        value["extra_info"].update(domain=row["domain"], split=row["mopd_split"],
                                   category=row.get("category", ""), task=row.get("task", ""))
        if self.qualification:
            value["extra_info"]["generation_max_tokens"] = min(256, row["generation_max_tokens"])
        return value
