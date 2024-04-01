<script setup lang="jsx">
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/vue'

let links = [
    { label: 'Home', url: '/', icon: 'material-symbols:house-outline' },
    { label: 'Preprocess', url: '/preprocess', icon: 'carbon:ibm-event-processing' },
    { label: 'Train', url: '/train', icon: 'carbon:machine-learning' },
    { label: 'Generate', url: '/generate', icon: 'material-symbols:music-note' },
]

const menu = {
    "Export": [
        { label: 'as MIDI', icon: 'mdi:export', action: exportAsMIDI },
        { label: 'as WAV', icon: 'mdi:export', exportAsWAV },
    ],
    "Misc.": [
        { label: 'About', icon: 'ri:information-2-line' },
    ]
}

function exportAsMIDI() {
    console.log('Exporting as MIDI...')
}

function exportAsWAV() {
    console.log('Exporting as WAV...')
}

function NavMenuLink(props, context) {
    return <MenuItem>
        <div class="w-full">
            <nuxt-link to={props.link.url}>
                <el-button class="w-full" style="justify-content: start;" text>
                    <icon class="mr-2" name={props.link.icon} size="1.5em" />
                    {props.link.label}
                </el-button>
            </nuxt-link>
        </div>
    </MenuItem>
}

function NavMenuItem(props, context) {
    return <MenuItem>
        <el-button onClick={() => {
            if (!props.item.action) return
            props.item.action()
        }} text>
            <icon class="mr-2" name={props.item.icon} size="1.5em" />
            {props.item.label}
        </el-button>
    </MenuItem>
}
</script>

<template>
    <div id="app-header" class="w-full z-10 h-fit relative py-2 px-8 bg-gray-50 shadow-md">

        <div class="flex flex-row justify-between items-center">
            <!-- App title -->
            <h1 class="select-none">SatAi 2.0</h1>

            <!-- Menu -->
            <Menu as="div" class="relative" v-slot="{ open }">

                <!-- Menu trigger -->
                <MenuButton class="relative flex rounded-full text-sm">
                    <el-button size="large" circle text :bg="open" style="padding: 1.6em;">
                        <icon v-if="!open" name="ic:menu" size="2em" />
                        <icon v-else name="material-symbols:close" size="1.5em" />
                    </el-button>
                </MenuButton>

                <!-- Menu items -->
                <transition enter-active-class="transition ease-out duration-100"
                    enter-from-class="transform opacity-0 scale-95" enter-to-class="transform opacity-100 scale-100"
                    leave-active-class="transition ease-in duration-75"
                    leave-from-class="transform opacity-100 scale-100" leave-to-class="transform opacity-0 scale-95">
                    <MenuItems
                        class="absolute menuitems flex flex-col items-stretch right-0 z-10 mt-2 min-w-48 origin-top-right rounded-md bg-white p-2 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">

                        <p class="px-4 py-1 text-xs text-gray-400 uppercase cursor-default">
                            Navigation
                        </p>
                        <NavMenuLink v-for="link of links" :link />

                        <template v-for="(sectionKey, index) of Object.keys(menu)" :key="index">
                            <p
                                :class="[index > 0 && 'mt-3', 'px-4 py-1 text-xs text-gray-400 uppercase cursor-default']">
                                {{ sectionKey }}
                            </p>
                            <NavMenuItem v-for="item of menu[sectionKey]" :item />
                        </template>
                    </MenuItems>
                </transition>

            </Menu>
        </div>
    </div>
</template>

<style scoped>
.menuitems .el-button+.el-button {
    margin-left: 0;
}

.el-button {
    justify-content: stretch;
}
</style>