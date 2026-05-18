<script setup lang="ts"></script>
<template>
    <button
        class="rounded-lg border border-base-300 px-3 py-2 transition hover:scale-95 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:scale-100"
    >
        <slot />
    </button>
</template>
