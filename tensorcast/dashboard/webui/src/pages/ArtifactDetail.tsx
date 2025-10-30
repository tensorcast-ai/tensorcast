import { useParams } from 'react-router-dom';

export default function ArtifactDetail() {
  const { artifactId } = useParams();
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Artifact: {artifactId}</h1>
      <div className="text-muted-foreground">展示副本列表、视图元数据、叶子覆盖与分片位置（稍后接入 API）。</div>
    </div>
  );
}


