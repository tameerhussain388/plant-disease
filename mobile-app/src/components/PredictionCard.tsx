import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { PredictionResult } from '../services/tfliteService';

interface Props {
  prediction: PredictionResult;
  rank: number;
}

export const PredictionCard: React.FC<Props> = ({ prediction, rank }) => {
  const isHealthy = prediction.label.toLowerCase().includes('healthy');
  const barColor = isHealthy ? '#2e7d32' : rank === 1 ? '#d32f2f' : '#f57c00';

  return (
    <View style={styles.card}>
      <View style={styles.headerRow}>
        <Text style={styles.rankBadge}>#{rank}</Text>
        <Text style={styles.label} numberOfLines={1}>{prediction.label}</Text>
        <Text style={styles.confidence}>{prediction.percentage}</Text>
      </View>
      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${Math.min(prediction.confidence * 100, 100)}%`, backgroundColor: barColor }]} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#ffffff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  rankBadge: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#666',
    marginRight: 8,
  },
  label: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#212121',
  },
  confidence: {
    fontSize: 14,
    fontWeight: '700',
    color: '#333333',
    marginLeft: 8,
  },
  progressTrack: {
    height: 8,
    backgroundColor: '#eeeeee',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
});
