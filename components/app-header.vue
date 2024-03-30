<script setup lang="jsx">
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/vue'

const menu = {
    "Export": [
        { label: 'as MIDI', icon: 'mdi:export-variant' },
        { label: 'as WAV', icon: 'mdi:export-variant' },
    ],
    "Misc.": [
        { label: 'About', icon: 'ri:information-2-line' },
    ]
}

function NavMenuItem(props, context) {
    return <MenuItem>
        <el-button text>
            <icon class="mr-2" name={props.item.icon} size="1.5em" />
            { props.item.label }
        </el-button>
    </MenuItem>
}
</script>

<template>
    <div id="app-header" class="w-full z-10 h-fit relative py-2 px-8 bg-gray-50 shadow-md">
        <div class="flex flex-row justify-between items-center">
            <!-- App Title -->
            <h1 class="select-none">SatAi 2.0</h1>

            <!-- Menu -->
            <Menu as="div" class="relative" v-slot="{ open }">

                <!-- Menu Trigger -->
                <MenuButton class="relative flex rounded-full text-sm">
                    <el-button size="large" circle text :bg="open" style="padding: 1.6em;">
                        <icon v-if="!open" name="ic:menu" size="2em" />
                        <icon v-else name="material-symbols:close" size="1.5em" />
                    </el-button>
                </MenuButton>

                <!-- Menu Items -->
                <transition enter-active-class="transition ease-out duration-100" enter-from-class="transform opacity-0 scale-95" enter-to-class="transform opacity-100 scale-100" leave-active-class="transition ease-in duration-75" leave-from-class="transform opacity-100 scale-100" leave-to-class="transform opacity-0 scale-95">
                    <MenuItems class="absolute menuitems flex flex-col items-stretch right-0 z-10 mt-2 min-w-48 origin-top-right rounded-md bg-white p-2 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                        <template v-for="(sectionKey, index) of Object.keys(menu)" :key="index">
                            <p :class="[index > 0 && 'mt-3', 'px-4 py-1 text-xs text-gray-400 uppercase cursor-default']">
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

.menuitems .el-button {
    justify-content: start;
}
</style>