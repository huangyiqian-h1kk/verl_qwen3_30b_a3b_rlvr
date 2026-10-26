"""Keep source rows untouched and use their stable IDs in validation journals."""

from multidomain.dataset import MultiDomainDataset


class TeacherDataset(MultiDomainDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item["uid"] = row["sample_id"]
        item["extra_info"].update({k: row.get(k, "") for k in ("category", "task")})
        return item
